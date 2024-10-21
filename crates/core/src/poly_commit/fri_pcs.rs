// Copyright 2024 Ulvetanna Inc.

use super::ring_switch::{evaluate_ring_switch_eq_ind, reduce_tensor_claim};
use crate::{
	challenger::{CanObserve, CanSample, CanSampleBits},
	composition::BivariateProduct,
	merkle_tree::VectorCommitScheme,
	poly_commit::{ring_switch::ring_switch_eq_ind_partial_eval, PolyCommitScheme},
	polynomial::{Error as PolynomialError, MultilinearExtension, MultilinearQuery},
	protocols::{
		fri::{self, FRIFolder, FRIParams, FRIVerifier, FoldRoundOutput},
		sumcheck_v2::{
			self, immediate_switchover_heuristic,
			prove::{RegularSumcheckProver, SumcheckProver},
			verify::interpolate_round_proof,
			RoundProof, SumcheckClaim,
		},
	},
	reed_solomon::reed_solomon::ReedSolomonCode,
	tensor_algebra::TensorAlgebra,
};
use binius_field::{
	packed::iter_packed_slice, BinaryField, ExtensionField, Field, PackedExtension, PackedField,
	PackedFieldIndexable,
};
use binius_hal::ComputationBackend;
use binius_math::univariate::EvaluationDomainFactory;
use binius_ntt::NTTOptions;
use binius_utils::{bail, checked_arithmetics::checked_log_2};
use std::{iter::repeat_with, marker::PhantomData, mem, ops::Deref};

/// The small-field FRI-based PCS from [DP24], also known as the FRI-Binius PCS.
///
/// This implements the Construction 6.1 in [DP24]. The polynomial is committed by committing the
/// Reed–Solomon encoding of its packed values with a vector commitment scheme (i.e. Merkle tree).
/// Evaluation proofs consist of interleaved sumcheck and FRI protocol invocations.
///
/// ## Type parameters
///
/// * `F` - the coefficient subfield
/// * `FDomain` - the field containing the sumcheck evaluation domains
/// * `FEncode` - a small field that is the Reed–Solomon alphabet
/// * `PE` - a packed extension field of `F` that is cryptographically big
/// * `DomainFactory` - a domain factory for the sumcheck reduction
/// * `VCS` - the vector commitment scheme used to commit the IOP oracles
///
/// [DP24]: <https://eprint.iacr.org/2024/504>
#[derive(Debug)]
pub struct FRIPCS<F, FDomain, FEncode, PE, DomainFactory, VCS>
where
	F: Field,
	FDomain: Field,
	FEncode: BinaryField,
	PE: PackedField + PackedExtension<FEncode>,
	PE::Scalar: BinaryField
		+ ExtensionField<F>
		+ PackedField<Scalar = PE::Scalar>
		+ ExtensionField<FDomain>
		+ ExtensionField<FEncode>,
{
	fri_params: FRIParams<PE::Scalar, FEncode, VCS>,
	/// Reed–Solomon code used for encoding during the commitment phase.
	// NB: This is the same RS code as `params.rs_code()` but is parameterized over a field instead
	// of a packed field for silly API-compliance reasons. We should refactor `fri_iopp` and
	// possibly LinearCode to handle both the proving and verification cases with the same RS code.
	rs_encoder: ReedSolomonCode<<PE as PackedExtension<FEncode>>::PackedSubfield>,
	domain_factory: DomainFactory,
	_marker: PhantomData<(F, FDomain, PE)>,
}

impl<F, FDomain, FEncode, FExt, PE, DomainFactory, VCS>
	FRIPCS<F, FDomain, FEncode, PE, DomainFactory, VCS>
where
	F: Field,
	FDomain: Field,
	FEncode: BinaryField,
	FExt: BinaryField
		+ PackedField<Scalar = FExt>
		+ ExtensionField<F>
		+ ExtensionField<FDomain>
		+ ExtensionField<FEncode>,
	PE: PackedFieldIndexable<Scalar = FExt>
		+ PackedExtension<FEncode, PackedSubfield: PackedFieldIndexable>,
	VCS: VectorCommitScheme<FExt> + Sync,
	VCS::Committed: Send + Sync,
{
	pub fn new(
		n_vars: usize,
		log_inv_rate: usize,
		fold_arities: &[usize],
		security_bits: usize,
		vcs_factory: impl Fn(usize) -> VCS,
		domain_factory: DomainFactory,
		ntt_options: NTTOptions,
	) -> Result<Self, Error> {
		let kappa = checked_log_2(<FExt as ExtensionField<F>>::DEGREE);

		// The number of variables of the packed polynomial.
		let n_packed_vars = n_vars
			.checked_sub(kappa)
			.ok_or(Error::IncorrectPolynomialSize { expected: kappa })?;

		if fold_arities.iter().sum::<usize>() > n_packed_vars {
			bail!(fri::Error::InvalidFoldAritySequence);
		}
		for &arity in fold_arities.iter() {
			if arity == 0 {
				bail!(fri::Error::FoldArityIsZero { index: 0 });
			}
		}

		// Choose the interleaved code batch size to align with the first fold arity, which is
		// optimal.
		let log_batch_size = fold_arities.first().copied().unwrap_or(0);
		let log_dim = n_packed_vars - log_batch_size;
		let log_len = log_dim + log_inv_rate;

		// TODO: set merkle cap heights correctly
		let poly_vcs = vcs_factory(log_len);

		let round_vcss = fold_arities
			.iter()
			.scan(log_len + log_batch_size, |len, arity| {
				*len -= arity;
				Some(vcs_factory(*len))
			})
			.collect();

		let rs_code = ReedSolomonCode::new(log_dim, log_inv_rate, NTTOptions::default())?;
		let n_test_queries = fri::calculate_n_test_queries::<FExt, _>(security_bits, &rs_code)?;
		let fri_params =
			FRIParams::new(rs_code, log_batch_size, poly_vcs, round_vcss, n_test_queries)?;
		let rs_encoder = ReedSolomonCode::new(log_dim, log_inv_rate, ntt_options)?;

		Ok(Self {
			fri_params,
			rs_encoder,
			domain_factory,
			_marker: PhantomData,
		})
	}

	/// Returns $\kappa$, the base-2 logarithm of the extension degree.
	pub const fn kappa() -> usize {
		<TensorAlgebra<F, PE::Scalar>>::kappa()
	}

	fn prove_interleaved_fri_sumcheck<Prover, Challenger>(
		&self,
		sumcheck_eval: TensorAlgebra<F, FExt>,
		codeword: &[PE],
		committed: &VCS::Committed,
		mut sumcheck_prover: Prover,
		mut challenger: Challenger,
	) -> Result<Proof<F, FExt, VCS>, Error>
	where
		Prover: SumcheckProver<FExt>,
		Challenger:
			CanObserve<FExt> + CanObserve<VCS::Commitment> + CanSample<FExt> + CanSampleBits<usize>,
	{
		let n_rounds = sumcheck_prover.n_vars();

		let mut fri_prover =
			FRIFolder::new(&self.fri_params, PE::unpack_scalars(codeword), committed)?;
		let mut rounds = Vec::with_capacity(n_rounds);
		let mut fri_commitments = Vec::with_capacity(self.fri_params.n_oracles());
		for _ in 0..n_rounds {
			let round_coeffs = sumcheck_prover.execute(FExt::ONE)?;
			let round_proof = round_coeffs.truncate();
			challenger.observe_slice(round_proof.coeffs());
			rounds.push(round_proof);

			let challenge = challenger.sample();

			match fri_prover.execute_fold_round(challenge)? {
				FoldRoundOutput::NoCommitment => {}
				FoldRoundOutput::Commitment(round_commitment) => {
					challenger.observe(round_commitment.clone());
					fri_commitments.push(round_commitment);
				}
			}

			sumcheck_prover.fold(challenge)?;
		}

		let _ = sumcheck_prover.finish()?;

		let (terminate_codeword, query_prover) = fri_prover.finalize()?;

		let fri_query_proofs = repeat_with(|| {
			let index = challenger.sample_bits(self.fri_params.rs_code().log_len());
			query_prover.prove_query(index)
		})
		.take(self.fri_params.n_test_queries())
		.collect::<Result<Vec<_>, _>>()?;

		Ok(Proof {
			sumcheck_eval,
			sumcheck_rounds: rounds,
			fri_commitments,
			fri_terminate_codeword: terminate_codeword,
			fri_query_proofs,
		})
	}

	#[allow(clippy::too_many_arguments)]
	fn verify_interleaved_fri_sumcheck<Challenger>(
		&self,
		claim: &SumcheckClaim<FExt, BivariateProduct>,
		codeword_commitment: &VCS::Commitment,
		sumcheck_round_proofs: Vec<RoundProof<FExt>>,
		fri_commitments: Vec<VCS::Commitment>,
		fri_terminate_codeword: fri::TerminateCodeword<FExt>,
		fri_query_proofs: Vec<fri::QueryProof<FExt, VCS::Proof>>,
		ring_switch_evaluator: impl FnOnce(&[FExt]) -> FExt,
		mut challenger: Challenger,
	) -> Result<(), Error>
	where
		Challenger:
			CanObserve<FExt> + CanObserve<VCS::Commitment> + CanSample<FExt> + CanSampleBits<usize>,
	{
		let n_rounds = claim.n_vars();
		if sumcheck_round_proofs.len() != n_rounds {
			return Err(VerificationError::Sumcheck(
				sumcheck_v2::VerificationError::NumberOfRounds,
			)
			.into());
		}

		if fri_commitments.len() != self.fri_params.n_oracles() {
			return Err(VerificationError::IncorrectNumberOfFRICommitments.into());
		}

		let mut round_vcs_iter = self.fri_params.round_vcss().iter().peekable();
		let mut fri_comm_iter = fri_commitments.iter().cloned();
		let log_inv_rate = self.fri_params.rs_code().log_inv_rate();

		assert_eq!(claim.composite_sums().len(), 1);
		let mut sum = claim.composite_sums()[0].sum;
		let mut challenges = Vec::with_capacity(n_rounds);
		for (round_no, round_proof) in sumcheck_round_proofs.into_iter().enumerate() {
			let n_vars = n_rounds - round_no;

			if round_proof.coeffs().len() != claim.max_individual_degree() {
				return Err(VerificationError::Sumcheck(
					sumcheck_v2::VerificationError::NumberOfCoefficients {
						round: round_no,
						expected: claim.max_individual_degree(),
					},
				)
				.into());
			}

			challenger.observe_slice(round_proof.coeffs());

			let challenge = challenger.sample();
			challenges.push(challenge);

			let observe_fri_comm = round_vcs_iter
				.peek()
				.map(|next_vcs| next_vcs.vector_len() == 1 << (log_inv_rate + n_vars - 1))
				.unwrap_or(false);
			if observe_fri_comm {
				let _ = round_vcs_iter.next();
				let comm = fri_comm_iter.next().expect(
					"round_vcss and fri_commitments lengths were checked to be equal; \
					iterators are incremented in lockstep; thus value must be Some",
				);
				challenger.observe(comm);
			}

			sum = interpolate_round_proof(round_proof, sum, challenge);
		}
		let verifier = FRIVerifier::new(
			&self.fri_params,
			codeword_commitment,
			&fri_commitments,
			&challenges,
			fri_terminate_codeword,
		)?;

		let ring_switch_eval = ring_switch_evaluator(&challenges);
		let final_fri_value = verifier.verify_last_oracle()?;
		if final_fri_value * ring_switch_eval != sum {
			return Err(VerificationError::IncorrectSumcheckEvaluation.into());
		}

		if fri_query_proofs.len() != self.fri_params.n_test_queries() {
			return Err(VerificationError::IncorrectNumberOfFRIQueries.into());
		}
		for query_proof in fri_query_proofs {
			let index = challenger.sample_bits(self.fri_params.rs_code().log_len());
			verifier.verify_query(index, query_proof)?;
		}

		Ok(())
	}
}

impl<F, FDomain, FEncode, FExt, P, PE, DomainFactory, VCS> PolyCommitScheme<P, FExt>
	for FRIPCS<F, FDomain, FEncode, PE, DomainFactory, VCS>
where
	F: Field,
	FDomain: Field,
	FEncode: BinaryField,
	FExt: BinaryField
		+ PackedField<Scalar = FExt>
		+ ExtensionField<F>
		+ ExtensionField<FDomain>
		+ ExtensionField<FEncode>
		+ PackedExtension<F>
		+ PackedExtension<FEncode>,
	P: PackedField<Scalar = F>,
	PE: PackedFieldIndexable<Scalar = FExt>
		+ PackedExtension<F, PackedSubfield = P>
		+ PackedExtension<FDomain>
		+ PackedExtension<FEncode, PackedSubfield: PackedFieldIndexable>,
	DomainFactory: EvaluationDomainFactory<FDomain>,
	VCS: VectorCommitScheme<FExt> + Sync,
	VCS::Committed: Send + Sync,
{
	type Commitment = VCS::Commitment;
	// Committed data is a tuple with the underlying codeword and the VCS committed data (ie.
	// Merkle internal node hashes).
	type Committed = (Vec<PE>, VCS::Committed);
	type Proof = Proof<F, FExt, VCS>;
	type Error = Error;

	fn n_vars(&self) -> usize {
		self.fri_params.n_fold_rounds() + Self::kappa()
	}

	fn commit<Data>(
		&self,
		polys: &[MultilinearExtension<P, Data>],
	) -> Result<(Self::Commitment, Self::Committed), Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
	{
		if polys.len() != 1 {
			todo!("handle batches of size greater than 1");
		}

		let poly = &polys[0];
		if poly.n_vars() != self.n_vars() {
			return Err(Error::IncorrectPolynomialSize {
				expected: self.n_vars(),
			});
		}
		let packed_evals = <PE as PackedExtension<F>>::cast_exts(poly.evals());

		let fri::CommitOutput {
			commitment,
			committed,
			codeword,
		} = fri::commit_interleaved(
			&self.rs_encoder,
			self.fri_params.log_batch_size(),
			self.fri_params.codeword_vcs(),
			packed_evals,
		)?;

		Ok((commitment, (codeword, committed)))
	}

	// Clippy allow is due to bug: https://github.com/rust-lang/rust-clippy/pull/12892
	#[allow(clippy::needless_borrows_for_generic_args)]
	fn prove_evaluation<Data, Challenger, Backend>(
		&self,
		challenger: &mut Challenger,
		committed: &Self::Committed,
		polys: &[MultilinearExtension<P, Data>],
		query: &[PE::Scalar],
		backend: Backend,
	) -> Result<Self::Proof, Self::Error>
	where
		Data: Deref<Target = [P]> + Send + Sync,
		Challenger: CanObserve<PE::Scalar>
			+ CanObserve<Self::Commitment>
			+ CanSample<PE::Scalar>
			+ CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}
		if polys.len() != 1 {
			todo!("handle batches of size greater than 1");
		}

		let poly = &polys[0];
		let packed_poly = MultilinearExtension::from_values_slice(
			<PE as PackedExtension<F>>::cast_exts(poly.evals()),
		)?;

		let (_, query_from_kappa) = query.split_at(Self::kappa());

		let expanded_query =
			MultilinearQuery::<PE, _>::with_full_query(query_from_kappa, backend.clone())?;
		let partial_eval = poly.evaluate_partial_high(&expanded_query)?;
		let sumcheck_eval =
			TensorAlgebra::<F, _>::new(iter_packed_slice(partial_eval.evals()).collect());

		challenger.observe_slice(sumcheck_eval.vertical_elems());

		// The challenges used to mix the rows of the tensor algebra coefficients.
		let tensor_mixing_challenges = challenger.sample_vec(Self::kappa());

		let sumcheck_claim = reduce_tensor_claim(
			self.n_vars(),
			sumcheck_eval.clone(),
			&tensor_mixing_challenges,
			backend.clone(),
		);
		let val = ring_switch_eq_ind_partial_eval::<F, _, _, _>(
			query_from_kappa,
			&tensor_mixing_challenges,
			backend.clone(),
		)?;
		let transparent = MultilinearExtension::from_values_generic(val)?;
		let sumcheck_prover = RegularSumcheckProver::new(
			vec![
				MultilinearExtension::<PE, _>::specialize::<PE>(packed_poly.to_ref()),
				MultilinearExtension::<PE, _>::specialize::<PE>(transparent.to_ref()),
			],
			sumcheck_claim.composite_sums().iter().cloned(),
			&self.domain_factory,
			immediate_switchover_heuristic,
			backend.clone(),
		)?;

		let (codeword, vcs_committed) = committed;
		self.prove_interleaved_fri_sumcheck(
			sumcheck_eval,
			codeword,
			vcs_committed,
			sumcheck_prover,
			challenger,
		)
	}

	fn verify_evaluation<Challenger, Backend>(
		&self,
		challenger: &mut Challenger,
		commitment: &Self::Commitment,
		query: &[FExt],
		proof: Self::Proof,
		values: &[FExt],
		backend: Backend,
	) -> Result<(), Self::Error>
	where
		Challenger: CanObserve<FExt>
			+ CanObserve<Self::Commitment>
			+ CanSample<FExt>
			+ CanSampleBits<usize>,
		Backend: ComputationBackend,
	{
		if query.len() != self.n_vars() {
			return Err(PolynomialError::IncorrectQuerySize {
				expected: self.n_vars(),
			}
			.into());
		}
		if values.len() != 1 {
			todo!("handle batches of size greater than 1");
		}

		let Proof {
			// This is s₀ in Protocol 4.1
			sumcheck_eval,
			sumcheck_rounds,
			fri_commitments,
			fri_terminate_codeword,
			fri_query_proofs,
		} = proof;

		let n_rounds = self.n_vars() - Self::kappa();
		assert!(n_rounds > 0, "this is checked in the constructor");

		let (query_to_kappa, query_from_kappa) = query.split_at(Self::kappa());

		challenger.observe_slice(sumcheck_eval.vertical_elems());

		// Check that the claimed sum is consistent with the tensor algebra element received.
		let expanded_query =
			MultilinearQuery::<FExt, _>::with_full_query(query_to_kappa, backend.clone())?;
		let computed_eval =
			MultilinearExtension::from_values_slice(sumcheck_eval.vertical_elems())?
				.evaluate(&expanded_query)?;
		if values[0] != computed_eval {
			return Err(VerificationError::IncorrectEvaluation.into());
		}

		// The challenges used to mix the rows of the tensor algebra coefficients.
		let tensor_mixing_challenges = challenger.sample_vec(Self::kappa());

		let sumcheck_claim = reduce_tensor_claim(
			self.n_vars(),
			sumcheck_eval,
			&tensor_mixing_challenges,
			backend.clone(),
		);

		self.verify_interleaved_fri_sumcheck(
			&sumcheck_claim,
			commitment,
			sumcheck_rounds,
			fri_commitments,
			fri_terminate_codeword,
			fri_query_proofs,
			|challenges| {
				evaluate_ring_switch_eq_ind::<F, _, _>(
					query_from_kappa,
					challenges,
					&tensor_mixing_challenges,
					backend.clone(),
				)
			},
			challenger,
		)
	}

	fn proof_size(&self, n_polys: usize) -> usize {
		if n_polys != 1 {
			todo!("handle batches of size greater than 1");
		}

		// TODO: This needs to get updated for higher-arity folding
		let fe_size = mem::size_of::<FExt>();
		let vc_size = mem::size_of::<VCS::Commitment>();

		let fri_termination_log_len =
			self.fri_params.n_final_challenges() + self.fri_params.rs_code().log_inv_rate();

		let sumcheck_eval_size = <TensorAlgebra<F, FExt>>::byte_size();
		// The number of rounds of sumcheck is equal to the number of variables minus kappa.
		// The function $h$ that we are sumchecking is multiquadratic, hence each round polynomial
		// is quadratic. We only send two of the three coefficients of this polynomial in `RoundProof`.
		let sumcheck_rounds_size = fe_size * 2 * (self.n_vars() - Self::kappa());
		// The number of FRI-commitments is encoded in state as `self.round_vcss.len()`. Alternatively, it is simply
		// `sumcheck_rounds_size - 1`.
		let fri_commitments_size = vc_size * (sumcheck_rounds_size - 1);
		// The length of the fri_terminate_codeword_size is just vector length of the last VCS .
		let fri_terminate_codeword_size = fe_size * (1 << fri_termination_log_len);
		// fri_query_proofs consists of n_test_queries of `QueryProof`.
		// each `QueryProof` consists of a number of `QueryRoundProof`s, This number is the number of
		// FRI-folds the verifier "receives".
		// for arity = 1, the number of FRI-folds the verifier receives is  which is `round_vcss.len()+1`
		// and the size of the coset is 2.
		let len_round_vcss = self.fri_params.rs_code().log_len() - fri_termination_log_len;
		let fri_query_proofs_size =
			(vc_size + 2 * fe_size) * (len_round_vcss + 1) * self.fri_params.n_test_queries();

		sumcheck_eval_size
			+ sumcheck_rounds_size
			+ fri_commitments_size
			+ fri_terminate_codeword_size
			+ fri_query_proofs_size
	}
}

/// A [`FRIPCS`] proof.
#[derive(Debug, Clone)]
pub struct Proof<F, FExt, VCS>
where
	F: Field,
	FExt: ExtensionField<F>,
	VCS: VectorCommitScheme<FExt>,
{
	sumcheck_eval: TensorAlgebra<F, FExt>,
	sumcheck_rounds: Vec<RoundProof<FExt>>,
	fri_commitments: Vec<VCS::Commitment>,
	fri_terminate_codeword: fri::TerminateCodeword<FExt>,
	fri_query_proofs: Vec<fri::QueryProof<FExt, VCS::Proof>>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("the polynomial must have {expected} variables")]
	IncorrectPolynomialSize { expected: usize },
	#[error("sumcheck error: {0}")]
	Sumcheck(#[from] sumcheck_v2::Error),
	#[error("polynomial error: {0}")]
	Polynomial(#[from] PolynomialError),
	#[error("FRI error: {0}")]
	FRI(#[from] fri::Error),
	#[error("NTT error: {0}")]
	NTT(#[from] binius_ntt::Error),
	#[error("verification failure: {0}")]
	Verification(#[from] VerificationError),
}

#[derive(Debug, thiserror::Error)]
pub enum VerificationError {
	#[error("sumcheck verification error: {0}")]
	Sumcheck(#[from] sumcheck_v2::VerificationError),
	#[error("evaluation value is inconsistent with the tensor evaluation")]
	IncorrectEvaluation,
	#[error("sumcheck final evaluation is incorrect")]
	IncorrectSumcheckEvaluation,
	#[error("incorrect number of FRI commitments")]
	IncorrectNumberOfFRICommitments,
	#[error("incorrect number of FRI query proofs")]
	IncorrectNumberOfFRIQueries,
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{challenger::new_hasher_challenger, merkle_tree::MerkleTreeVCS};
	use binius_field::{
		arch::packed_polyval_128::PackedBinaryPolyval1x128b,
		as_packed_field::{PackScalar, PackedType},
		underlier::{Divisible, UnderlierType, WithUnderlier},
		BinaryField128b, BinaryField16b, BinaryField1b, BinaryField32b, BinaryField8b,
	};
	use binius_hal::make_portable_backend;
	use binius_hash::{GroestlDigestCompression, GroestlHasher};
	use binius_math::IsomorphicEvaluationDomainFactory;
	use rand::{prelude::StdRng, SeedableRng};

	fn test_commit_prove_verify_success<U, F, FA, FE>(
		n_vars: usize,
		log_inv_rate: usize,
		fold_arities: &[usize],
	) where
		U: UnderlierType
			+ PackScalar<F>
			+ PackScalar<FA>
			+ PackScalar<FE>
			+ PackScalar<BinaryField8b>
			+ Divisible<u8>,
		F: Field,
		FA: BinaryField,
		FE: BinaryField
			+ ExtensionField<F>
			+ ExtensionField<FA>
			+ ExtensionField<BinaryField8b>
			+ PackedField<Scalar = FE>
			+ PackedExtension<F>
			+ PackedExtension<FA, PackedSubfield: PackedFieldIndexable>
			+ PackedExtension<BinaryField8b, PackedSubfield: PackedFieldIndexable>,
		PackedType<U, FA>: PackedFieldIndexable,
		PackedType<U, FE>: PackedFieldIndexable,
	{
		let mut rng = StdRng::seed_from_u64(0);
		let backend = make_portable_backend();

		let multilin = MultilinearExtension::from_values(
			repeat_with(|| <PackedType<U, F>>::random(&mut rng))
				.take(1 << (n_vars - <PackedType<U, F>>::LOG_WIDTH))
				.collect(),
		)
		.unwrap();
		assert_eq!(multilin.n_vars(), n_vars);

		let eval_point = repeat_with(|| <FE as Field>::random(&mut rng))
			.take(n_vars)
			.collect::<Vec<_>>();

		let eval_query = MultilinearQuery::<FE, binius_hal::CpuBackend>::with_full_query(
			&eval_point,
			backend.clone(),
		)
		.unwrap();
		let eval = multilin.evaluate(&eval_query).unwrap();

		let make_merkle_vcs = |log_len| {
			MerkleTreeVCS::<FE, _, GroestlHasher<_>, _>::new(
				log_len,
				0,
				GroestlDigestCompression::default(),
			)
		};

		let domain_factory = IsomorphicEvaluationDomainFactory::<BinaryField8b>::default();
		let pcs = FRIPCS::<F, BinaryField8b, FA, PackedType<U, FE>, _, _>::new(
			n_vars,
			log_inv_rate,
			fold_arities,
			32,
			make_merkle_vcs,
			domain_factory,
			NTTOptions::default(),
		)
		.unwrap();

		let (commitment, committed) = pcs.commit(&[multilin.to_ref()]).unwrap();

		let mut challenger = new_hasher_challenger::<_, GroestlHasher<_>>();
		challenger.observe(commitment.clone());

		let mut prover_challenger = challenger.clone();
		let proof = pcs
			.prove_evaluation(
				&mut prover_challenger,
				&committed,
				&[multilin],
				&eval_point,
				backend.clone(),
			)
			.unwrap();

		let mut verifier_challenger = challenger.clone();
		pcs.verify_evaluation(
			&mut verifier_challenger,
			&commitment,
			&eval_point,
			proof,
			&[eval],
			backend.clone(),
		)
		.unwrap();
	}

	#[test]
	fn test_commit_prove_verify_success_1b_128b() {
		test_commit_prove_verify_success::<
			<PackedBinaryPolyval1x128b as WithUnderlier>::Underlier,
			BinaryField1b,
			BinaryField16b,
			BinaryField128b,
		>(18, 2, &[3, 3, 3]);
	}

	#[test]
	fn test_commit_prove_verify_success_32b_128b() {
		test_commit_prove_verify_success::<
			<PackedBinaryPolyval1x128b as WithUnderlier>::Underlier,
			BinaryField32b,
			BinaryField16b,
			BinaryField128b,
		>(12, 2, &[3, 3, 3]);
	}
}
