// Copyright 2024-2025 Irreducible Inc.

//! This module defines the `ConstraintSystemBuilder`, which helps construct
//! a constraint system with oracles, constraints, flush operations, and more.
//!
//! # Module Overview
//!
//! - [`ConstraintSystemBuilder`] is the primary struct.
//! - The builder can optionally hold a witness builder to track columns.
//! - It provides methods to flush data in certain directions (`send`, `receive`),
//!   assert zero checks, add oracles, and build a final [`ConstraintSystem`].
//!
//! # Example
//! ```rust
//! // Simple top-level usage example
//! use binius_circuits::builder::ConstraintSystemBuilder;
//! 
//! fn example() {
//!     let mut builder = ConstraintSystemBuilder::new();
//!     builder.assert_zero("example", vec![], binius_math::ArithExpr::from(0));
//!     let system = builder.build().unwrap();
//!     // Now you can use `system`...
//! }
//! ```

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use anyhow::{anyhow, ensure};
use binius_core::{
	constraint_system::{
		channel::{ChannelId, Flush, FlushDirection},
		ConstraintSystem,
	},
	oracle::{
		ConstraintSetBuilder, Error as OracleError, MultilinearOracleSet, OracleId,
		ProjectionVariant, ShiftVariant,
	},
	polynomial::MultivariatePoly,
	transparent::step_down::StepDown,
	witness::MultilinearExtensionIndex,
};
use binius_field::{as_packed_field::PackScalar, BinaryField1b};
use binius_math::ArithExpr;
use binius_utils::bail;

use crate::builder::{
	types::{F, U},
	witness,
};

/// A builder for creating a [`ConstraintSystem`] with optional witness columns,
/// multiple oracles, and zero-check assertions.
///
/// # Usage
///
/// Typically, you:
/// 1. Create a new builder via [`ConstraintSystemBuilder::new()`] or [`ConstraintSystemBuilder::new_with_witness()`].
/// 2. Add oracles, constraints, flush operations, and so on.
/// 3. Finally, call [`ConstraintSystemBuilder::build()`] to obtain a [`ConstraintSystem`].
///
/// # Example
/// ```rust
/// // 1) Create the builder
/// let mut builder = ConstraintSystemBuilder::new();
///
/// // 2) Use the builder to add constraints/oracles
/// builder.assert_zero("my_constraint", vec![], binius_math::ArithExpr::from(0));
///
/// // 3) Build the final system
/// let system = builder.build().unwrap();
/// ```
#[derive(Default)]
pub struct ConstraintSystemBuilder<'arena> {
	oracles: Rc<RefCell<MultilinearOracleSet<F>>>,
	constraints: ConstraintSetBuilder<F>,
	non_zero_oracle_ids: Vec<OracleId>,
	flushes: Vec<Flush>,
	step_down_dedup: HashMap<(usize, usize), OracleId>,
	witness: Option<witness::Builder<'arena>>,
	next_channel_id: ChannelId,
	namespace_path: Vec<String>,
}

impl<'arena> ConstraintSystemBuilder<'arena> {

	/// Creates a new builder that does not track witness columns.
	///
	/// # Example
	/// ```rust
	/// // (1) Create the builder
	/// let mut builder = ConstraintSystemBuilder::new();
	/// // (2) Optionally add constraints here
	/// builder.assert_zero("demo", vec![], binius_math::ArithExpr::from(1));
	/// // (3) Build the system
	/// let system = builder.build().unwrap();
	/// ```
	pub fn new() -> Self {
		Self::default()
	}

	/// Creates a new builder with a witness builder allocated on the given `allocator`.
	///
	/// # Example
	/// ```rust
	/// // (1) Create an allocator and builder
	/// let bump = bumpalo::Bump::new();
	/// let mut builder = ConstraintSystemBuilder::new_with_witness(&bump);
	/// // (2) Use builder methods with witness tracking
	/// builder.assert_zero("with_witness", vec![], binius_math::ArithExpr::from(2));
	/// // (3) Build
	/// let system = builder.build().unwrap();
	/// ```
	pub fn new_with_witness(allocator: &'arena bumpalo::Bump) -> Self {
		let oracles = Rc::new(RefCell::new(MultilinearOracleSet::new()));
		Self {
			witness: Some(witness::Builder::new(allocator, oracles.clone())),
			oracles,
			..Default::default()
		}
	}

	/// Finalizes the builder and constructs the [`ConstraintSystem`].
	///
	/// # Errors
	/// If building fails for any reason, returns an [`anyhow::Error`].
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// // Some hypothetical usage
	/// builder.assert_zero("example", vec![], binius_math::ArithExpr::from(0));
	/// let system = builder.build().unwrap();
	/// println!("Built system with max channel ID: {}", system.max_channel_id);
	/// ```
	#[allow(clippy::type_complexity)]
	pub fn build(self) -> Result<ConstraintSystem<F>, anyhow::Error> {
		let table_constraints = self.constraints.build(&self.oracles.borrow())?;
		Ok(ConstraintSystem {
			max_channel_id: self
				.flushes
				.iter()
				.map(|flush| flush.channel_id)
				.max()
				.unwrap_or(0),
			table_constraints,
			non_zero_oracle_ids: self.non_zero_oracle_ids,
			oracles: Rc::into_inner(self.oracles)
				.ok_or_else(|| {
					anyhow!("Failed to build ConstraintSystem: references still exist to oracles")
				})?
				.into_inner(),
			flushes: self.flushes,
		})
	}

	/// Returns a mutable reference to the witness builder, if available.
	/// Returns `None` if the builder is in verifier mode or the witness was taken.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// // Typically `None` if we never used `new_with_witness`
	/// let maybe_witness = builder.witness();
	/// assert!(maybe_witness.is_none());
	/// ```
	pub const fn witness(&mut self) -> Option<&mut witness::Builder<'arena>> {
		self.witness.as_mut()
	}

	/// Consumes the witness builder and returns a [`MultilinearExtensionIndex`], if available.
	///
	/// # Errors
	/// Returns an error if the witness is missing or already taken.
	///
	/// # Example
	/// ```rust
	/// let bump = bumpalo::Bump::new();
	/// let mut builder = ConstraintSystemBuilder::new_with_witness(&bump);
	/// // ... add constraints ...
	/// let witness_index = builder.take_witness().unwrap();
	/// println!("We got a witness index for the circuit!");
	/// ```
	pub fn take_witness(
		&mut self,
	) -> Result<MultilinearExtensionIndex<'arena, U, F>, anyhow::Error> {
		Option::take(&mut self.witness)
			.ok_or_else(|| {
				anyhow!("Witness is missing. Are you in verifier mode, or have you already extraced the witness?")
			})?
			.build()
	}

	/// Flushes a set of oracles in a particular direction and channel with a given repetition count.
	///
	/// Internally, this calls [`flush_with_multiplicity`] with a default `multiplicity` of 1.
	///
	/// # Errors
	/// Fails if the flush is invalid (e.g., oracle dimensions mismatch).
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let chan_id = builder.add_channel();
	/// let some_oracle_id = builder.add_committed("my_oracle", 4, 0);
	/// builder.flush(binius_core::constraint_system::channel::FlushDirection::Push, chan_id, 1, vec![some_oracle_id]).unwrap();
	/// println!("Flushed oracles!");
	/// ```
	pub fn flush(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleId> + Clone,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		self.flush_with_multiplicity(direction, channel_id, count, oracle_ids, 1)
	}

	/// Similar to [`flush`], but allows specifying a `multiplicity` for repeated flush patterns.
	///
	/// # Errors
	/// Fails if the flush or oracle dimensions mismatch, or `step_down` cannot be created.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let chan_id = builder.add_channel();
	/// let oracle_id = builder.add_committed("my_oracle", 4, 0);
	/// builder.flush_with_multiplicity(binius_core::constraint_system::channel::FlushDirection::Pull, chan_id, 2, vec![oracle_id], 3).unwrap();
	/// println!("Flushed with multiplicity!");
	/// ```
	pub fn flush_with_multiplicity(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleId> + Clone,
		multiplicity: u64,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		let n_vars = self.log_rows(oracle_ids.clone())?;

		let selector = if let Some(&selector) = self.step_down_dedup.get(&(n_vars, count)) {
			selector
		} else {
			let step_down = StepDown::new(n_vars, count)?;
			let selector = self.add_transparent(
				format!("internal step_down {count}-{n_vars}"),
				step_down.clone(),
			)?;

			if let Some(witness) = self.witness() {
				step_down.populate(witness.new_column::<BinaryField1b>(selector).packed());
			}

			self.step_down_dedup.insert((n_vars, count), selector);
			selector
		};

		self.flush_custom(direction, channel_id, selector, oracle_ids, multiplicity)
	}

	/// Performs a custom flush with a user-specified `selector`.
	///
	/// # Errors
	/// Returns an error if the selector’s `n_vars` differs from the oracles’ `n_vars`.
	pub fn flush_custom(
		&mut self,
		direction: FlushDirection,
		channel_id: ChannelId,
		selector: OracleId,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		multiplicity: u64,
	) -> anyhow::Result<()> {
		let oracles = oracle_ids.into_iter().collect::<Vec<_>>();
		let log_rows = self.log_rows(oracles.iter().copied())?;
		ensure!(
			log_rows == self.log_rows([selector])?,
			"Selector {} n_vars does not match flush {:?}",
			selector,
			oracles
		);

		self.flushes.push(Flush {
			channel_id,
			direction,
			selector,
			oracles,
			multiplicity,
		});

		Ok(())
	}

	/// Pushes (sends) oracle data into the channel with a certain repetition count.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let cid = builder.add_channel();
	/// let oid = builder.add_committed("push_oracle", 3, 0);
	/// builder.send(cid, 1, vec![oid]).unwrap();
	/// println!("Send completed!");
	/// ```
	pub fn send(
		&mut self,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleId> + Clone,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		self.flush(FlushDirection::Push, channel_id, count, oracle_ids)
	}

	/// Pulls (receives) oracle data from the channel with a certain repetition count.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let cid = builder.add_channel();
	/// let oid = builder.add_committed("receive_oracle", 3, 0);
	/// builder.receive(cid, 1, vec![oid]).unwrap();
	/// println!("Receive completed!");
	/// ```
	pub fn receive(
		&mut self,
		channel_id: ChannelId,
		count: usize,
		oracle_ids: impl IntoIterator<Item = OracleId> + Clone,
	) -> anyhow::Result<()>
	where
		U: PackScalar<BinaryField1b>,
	{
		self.flush(FlushDirection::Pull, channel_id, count, oracle_ids)
	}

	/// Asserts that the provided [`ArithExpr`] evaluates to zero across the specified `oracle_ids`.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// builder.assert_zero("my_assert", vec![], binius_math::ArithExpr::from(42));
	/// // The expression is always 42, so it won't truly be zero, but this is just an example.
	/// ```
	pub fn assert_zero(
		&mut self,
		name: impl ToString,
		oracle_ids: impl IntoIterator<Item = OracleId>,
		composition: ArithExpr<F>,
	) {
		self.constraints
			.add_zerocheck(name, oracle_ids, composition);
	}

	/// Asserts that the specified `oracle_id` is non-zero. This is typically used
	/// to ensure that certain data is not identically zero.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let id = builder.add_committed("some_oracle", 4, 0);
	/// builder.assert_not_zero(id);
	/// println!("We want this oracle to be non-zero!");
	/// ```
	pub fn assert_not_zero(&mut self, oracle_id: OracleId) {
		self.non_zero_oracle_ids.push(oracle_id);
	}

	/// Creates and returns a new channel ID. Each channel ID is unique for this builder.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let cid = builder.add_channel();
	/// println!("Got channel ID: {}", cid);
	/// ```
	pub const fn add_channel(&mut self) -> ChannelId {
		let channel_id = self.next_channel_id;
		self.next_channel_id += 1;
		channel_id
	}

	/// Adds a committed oracle to the system with the given `n_vars` and `tower_level`.
	/// Returns the generated [`OracleId`].
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let my_oracle = builder.add_committed("example_oracle", 8, 0);
	/// println!("Oracle ID: {}", my_oracle);
	/// ```
	pub fn add_committed(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		tower_level: usize,
	) -> OracleId {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.committed(n_vars, tower_level)
	}

	/// Adds multiple committed oracles to the system and returns their IDs as an array.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let oracles = builder.add_committed_multiple::<2>("multi_oracle", 5, 0);
	/// println!("Two oracles: {:?} and {:?}", oracles[0], oracles[1]);
	/// ```
	pub fn add_committed_multiple<const N: usize>(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		tower_level: usize,
	) -> [OracleId; N] {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.committed_multiple(n_vars, tower_level)
	}

	/// Creates a new oracle that is a linear combination of existing oracles.
	///
	/// # Errors
	/// Returns [`OracleError`] if creation fails (e.g., dimension mismatches).
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let a = builder.add_committed("a", 4, 0);
	/// let b = builder.add_committed("b", 4, 0);
	/// let combo = builder.add_linear_combination("a_plus_b", 4, vec![(a, 1.into()), (b, 1.into())]).unwrap();
	/// println!("Created linear combination: {}", combo);
	/// ```
	pub fn add_linear_combination(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.linear_combination(n_vars, inner)
	}

	/// Similar to [`add_linear_combination`], but with an additional `offset` added to every value.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// let a = builder.add_committed("a", 4, 0);
	/// let offset = 10.into();
	/// let combo = builder.add_linear_combination_with_offset("with_offset", 4, offset, vec![(a, 2.into())]).unwrap();
	/// println!("Created linear combination with offset: {}", combo);
	/// ```
	pub fn add_linear_combination_with_offset(
		&mut self,
		name: impl ToString,
		n_vars: usize,
		offset: F,
		inner: impl IntoIterator<Item = (OracleId, F)>,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.linear_combination_with_offset(n_vars, offset, inner)
	}

	/// Creates a packed oracle from an existing oracle, potentially reducing the number of variables (`log_degree`).
	///
	/// # Errors
	/// Returns an error if the packing process is invalid for the given `log_degree`.
	pub fn add_packed(
		&mut self,
		name: impl ToString,
		id: OracleId,
		log_degree: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.packed(id, log_degree)
	}

	/// Creates a projected oracle from an existing oracle, applying a certain [`ProjectionVariant`].
	///
	/// # Errors
	/// Returns an error if the projection fails for any reason.
	pub fn add_projected(
		&mut self,
		name: impl ToString,
		id: OracleId,
		values: Vec<F>,
		variant: ProjectionVariant,
	) -> Result<usize, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.projected(id, values, variant)
	}

	/// Creates a repeating oracle from `id`, replicating it `2^log_count` times in some logical sense.
	///
	/// # Errors
	/// Returns an error if the repeating setup is invalid.
	pub fn add_repeating(
		&mut self,
		name: impl ToString,
		id: OracleId,
		log_count: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.repeating(id, log_count)
	}

	/// Creates a shifted oracle from an existing oracle with the specified offset, block bits, and shift variant.
	///
	/// # Errors
	/// Returns an error if shifting is invalid for the oracle’s dimension.
	pub fn add_shifted(
		&mut self,
		name: impl ToString,
		id: OracleId,
		offset: usize,
		block_bits: usize,
		variant: ShiftVariant,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.shifted(id, offset, block_bits, variant)
	}

	/// Adds a transparent polynomial oracle for the given multivariate polynomial.
	///
	/// # Errors
	/// Returns an error if it fails to register the polynomial.
	///
	/// # Example
	/// ```rust
	/// use binius_math::ArithExpr;
	/// 
	/// let mut builder = ConstraintSystemBuilder::new();
	/// // Suppose we have some custom polynomial that implements `MultivariatePoly<F>`.
	/// struct MyPoly;
	/// impl binius_core::polynomial::MultivariatePoly<binius_field::BinaryField1b> for MyPoly {
	///     // ...
	///     fn degree(&self) -> usize { 1 }
	///     fn evaluate(&self, _point: &[binius_field::BinaryField1b]) -> binius_field::BinaryField1b {
	///         binius_field::BinaryField1b::ONE
	///     }
	/// }
	/// 
	/// let poly_id = builder.add_transparent("custom_poly", MyPoly).unwrap();
	/// println!("Transparent poly ID: {}", poly_id);
	/// ```
	pub fn add_transparent(
		&mut self,
		name: impl ToString,
		poly: impl MultivariatePoly<F> + 'static,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.transparent(poly)
	}

	/// Creates an oracle that zero-pads another oracle up to `n_vars`.
	///
	/// # Errors
	/// Returns an error if zero-padding is not feasible for `id`.
	pub fn add_zero_padded(
		&mut self,
		name: impl ToString,
		id: OracleId,
		n_vars: usize,
	) -> Result<OracleId, OracleError> {
		self.oracles
			.borrow_mut()
			.add_named(self.scoped_name(name))
			.zero_padded(id, n_vars)
	}

	/// Qualifies a name with the current namespace path, if any.
	fn scoped_name(&self, name: impl ToString) -> String {
		let name = name.to_string();
		if self.namespace_path.is_empty() {
			name
		} else {
			format!("{}::{name}", self.namespace_path.join("::"))
		}
	}

	/// Pushes a string onto the namespace stack. Useful for grouping oracles by context.
	///
	/// # Example
	/// ```rust
	/// let mut builder = ConstraintSystemBuilder::new();
	/// builder.push_namespace("groupA");
	/// builder.add_committed("x", 10, 0);
	/// builder.push_namespace("subgroupB");
	/// builder.add_committed("y", 10, 0);
	/// builder.pop_namespace();
	/// builder.pop_namespace();
	/// ```
	pub fn push_namespace(&mut self, name: impl ToString) {
		self.namespace_path.push(name.to_string());
	}

	/// Pops the most recent namespace from the stack, if any.
	///
	/// # Example
	/// See [`push_namespace`].
	pub fn pop_namespace(&mut self) {
		self.namespace_path.pop();
	}

	/// Returns the number of rows (logarithm of the row count) shared by a set of columns.
	///
	/// # Errors
	/// Fails if no columns are provided or the columns have differing number of rows.
	pub fn log_rows(
		&self,
		oracle_ids: impl IntoIterator<Item = OracleId>,
	) -> anyhow::Result<usize> {
		let mut oracle_ids = oracle_ids.into_iter();
		let oracles = self.oracles.borrow();
		let Some(first_id) = oracle_ids.next() else {
			bail!(anyhow!("log_rows: You need to specify at least one column"));
		};
		let log_rows = oracles.n_vars(first_id);
		if oracle_ids.any(|id| oracles.n_vars(id) != log_rows) {
			bail!(anyhow!("log_rows: All columns must have the same number of rows"))
		}
		Ok(log_rows)
	}
}
