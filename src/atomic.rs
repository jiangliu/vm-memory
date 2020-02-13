// Copyright (C) 2019 Alibaba Cloud Computing. All rights reserved.
// Copyright (C) 2020 Red Hat, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A wrapper over an `ArcSwap<GuestMemory>` struct to support RCU-style mutability.
//!
//! With the `backend-atomic` feature enabled, simply replacing `GuestMemoryMmap`
//! with `GuestMemoryAtomic<GuestMemoryMmap>` will enable support for mutable memory maps.
//! To support mutable memory maps, devices will also need to use
//! `GuestAddressSpace::memory_map()` to gain temporary access to guest memory.

extern crate arc_swap;

use arc_swap::{ArcSwap, Guard};
use std::ops::Deref;
use std::result;
use std::sync::{Arc, LockResult, Mutex, MutexGuard, PoisonError};

use crate::{GuestAddressSpace, GuestMemory, GuestMemoryMut};

/// A fast implementation of a mutable collection of memory regions.
///
/// This implementation uses ArcSwap to provide RCU-like snapshotting of the memory map:
/// every update of the memory map creates a completely new GuestMemory object, and
/// readers will not be blocked because the copies they retrieved will be collected once
/// no one can access them anymore.  Under the assumption that updates to the memory map
/// are rare, this allows a very efficient implementation of the `memory_map()` method.
#[derive(Clone, Debug)]
pub struct GuestMemoryAtomic<M: GuestMemory> {
    // GuestAddressSpace<M>, which we want to implement, is basically a drop-in
    // replacement for &M.  Therefore, we need to pass to devices the GuestMemoryAtomic
    // rather than a reference to it.  To obtain this effect we wrap the actual fields
    // of GuestMemoryAtomic with an Arc, and derive the Clone trait.  See the
    // documentation for GuestAddressSpace for an example.
    inner: Arc<(ArcSwap<M>, Mutex<()>)>,
}

impl<M: GuestMemory> From<Arc<M>> for GuestMemoryAtomic<M> {
    /// create a new GuestMemoryAtomic object whose initial contents come from
    /// the `map` reference counted GuestMemory.
    fn from(map: Arc<M>) -> Self {
        let inner = (ArcSwap::new(map), Mutex::new(()));
        GuestMemoryAtomic {
            inner: Arc::new(inner),
        }
    }
}

impl<M: GuestMemory> GuestMemoryAtomic<M> {
    /// create a new GuestMemoryAtomic object whose initial contents come from
    /// the `map` GuestMemory.
    pub fn new(map: M) -> Self {
        Arc::new(map).into()
    }

    fn load(&self) -> Guard<'static, Arc<M>> {
        self.inner.0.load()
    }

    /// Acquires the update mutex for the GuestMemoryAtomic, blocking the current
    /// thread until it is able to do so.  The returned RAII guard allows for
    /// scoped unlock of the mutex (that is, the mutex will be unlocked when
    /// the guard goes out of scope), and optionally also for replacing the
    /// contents of the GuestMemoryAtomic when the lock is dropped.
    pub fn lock(&self) -> LockResult<GuestMemoryExclusiveGuard<M>> {
        match self.inner.1.lock() {
            Ok(guard) => Ok(GuestMemoryExclusiveGuard {
                parent: self,
                _guard: guard,
            }),
            Err(err) => Err(PoisonError::new(GuestMemoryExclusiveGuard {
                parent: self,
                _guard: err.into_inner(),
            })),
        }
    }
}

impl<M: GuestMemory> GuestAddressSpace<M> for GuestMemoryAtomic<M> {
    type T = GuestMemoryLoadGuard<M>;

    fn memory_map(&self) -> Self::T {
        GuestMemoryLoadGuard { guard: self.load() }
    }
}

impl<M: GuestMemory> GuestMemoryMut for GuestMemoryAtomic<M> {
    type R = <M as GuestMemoryMut>::R;

    fn with_regions<F, E>(&self, cb: F) -> result::Result<(), E>
    where
        F: Fn(usize, &Self::R) -> result::Result<(), E>,
    {
        self.load().with_regions(cb)
    }

    fn with_regions_mut<F, E>(&self, cb: F) -> result::Result<(), E>
    where
        F: FnMut(usize, &Self::R) -> result::Result<(), E>,
    {
        self.load().with_regions_mut(cb)
    }

    fn map_and_fold<F, G, T>(&self, init: T, mapf: F, foldf: G) -> T
    where
        F: Fn((usize, &Self::R)) -> T,
        G: Fn(T, T) -> T,
    {
        self.load().map_and_fold(init, mapf, foldf)
    }
}

/// A guard that provides temporary access to a GuestMemoryAtomic.  This
/// object is returned from the `memory_map()` method.  It dereference to
/// a snapshot of the GuestMemory, so it can be used transparently to
/// access memory.
#[derive(Debug)]
pub struct GuestMemoryLoadGuard<M: GuestMemory> {
    guard: Guard<'static, Arc<M>>,
}

impl<M: GuestMemory> GuestMemoryLoadGuard<M> {
    /// Make a clone of the held pointer and returns it.  This is more
    /// expensive than just using the snapshot, but it allows to hold on
    /// to the snapshot outside the scope of the guard.  It also allows
    /// writers to proceed, so it is recommended if the reference must
    /// be held for a long time (including for caching purposes).
    pub fn into_inner(self) -> Arc<M> {
        Guard::into_inner(self.guard)
    }
}

impl<M: GuestMemory> Deref for GuestMemoryLoadGuard<M> {
    type Target = M;

    fn deref(&self) -> &Self::Target {
        &*self.guard
    }
}

/// An RAII implementation of a "scoped lock" for GuestMemoryAtomic.  When
/// this structure is dropped (falls out of scope) the lock will be unlocked,
/// possibly after updating the memory map represented by the
/// GuestMemoryAtomic that created the guard.
pub struct GuestMemoryExclusiveGuard<'a, M: GuestMemory> {
    parent: &'a GuestMemoryAtomic<M>,
    _guard: MutexGuard<'a, ()>,
}

impl<M: GuestMemory> GuestMemoryExclusiveGuard<'_, M> {
    /// Replace the memory map in the GuestMemoryAtomic that created the guard
    /// with the new memory map, `map`.  The lock is then dropped since this
    /// method consumes the guard.
    pub fn replace(self, map: M) {
        self.parent.inner.0.store(Arc::new(map))
    }
}

#[cfg(test)]
#[cfg(feature = "backend-mmap")]
mod tests {
    use super::*;
    use crate::{
        GuestAddress, GuestMemory, GuestMemoryMmap, GuestMemoryRegion, GuestMemoryResult,
        GuestUsize,
    };

    type GuestMemoryMmapAtomic = GuestMemoryAtomic<GuestMemoryMmap>;

    #[test]
    fn test_atomic_basic() {
        let region_size = 0x400;
        let regions = vec![
            (GuestAddress(0x0), region_size),
            (GuestAddress(0x1000), region_size),
        ];
        let mut iterated_regions = Vec::new();
        let gmm = GuestMemoryMmap::from_ranges(&regions).unwrap();
        let gm = GuestMemoryMmapAtomic::new(gmm);

        let res: GuestMemoryResult<()> = gm.with_regions(|_, region| {
            assert_eq!(region.len(), region_size as GuestUsize);
            Ok(())
        });
        assert!(res.is_ok());
        let res: GuestMemoryResult<()> = gm.with_regions_mut(|_, region| {
            iterated_regions.push((region.start_addr(), region.len() as usize));
            Ok(())
        });
        assert!(res.is_ok());
        assert_eq!(regions, iterated_regions);

        assert!(regions
            .iter()
            .map(|x| (x.0, x.1))
            .eq(iterated_regions.iter().map(|x| *x)));
    }

    #[test]
    fn test_atomic_memory_map() {
        let region_size = 0x400;
        let regions = vec![
            (GuestAddress(0x0), region_size),
            (GuestAddress(0x1000), region_size),
        ];
        let mut iterated_regions = Vec::new();
        let gmm = GuestMemoryMmap::from_ranges(&regions).unwrap();
        let gm = GuestMemoryMmapAtomic::new(gmm);
        let snapshot = gm.memory_map();

        let res: GuestMemoryResult<()> = snapshot.with_regions(|_, region| {
            assert_eq!(region.len(), region_size as GuestUsize);
            Ok(())
        });
        assert!(res.is_ok());
        let res: GuestMemoryResult<()> = snapshot.with_regions_mut(|_, region| {
            iterated_regions.push((region.start_addr(), region.len() as usize));
            Ok(())
        });
        assert!(res.is_ok());
        assert_eq!(regions, iterated_regions);
        assert_eq!(snapshot.num_regions(), 2);
        assert!(snapshot.find_region(GuestAddress(0x1000)).is_some());
        assert!(snapshot.find_region(GuestAddress(0x10000)).is_none());

        assert!(regions
            .iter()
            .map(|x| (x.0, x.1))
            .eq(iterated_regions.iter().map(|x| *x)));

        let snapshot2 = snapshot.into_inner();
        let res: GuestMemoryResult<()> = snapshot2.with_regions(|_, region| {
            assert_eq!(region.len(), region_size as GuestUsize);
            Ok(())
        });
        assert!(res.is_ok());
        let res: GuestMemoryResult<()> = snapshot2.with_regions_mut(|_, _| Ok(()));
        assert!(res.is_ok());
        assert_eq!(snapshot2.num_regions(), 2);
        assert!(snapshot2.find_region(GuestAddress(0x1000)).is_some());
        assert!(snapshot2.find_region(GuestAddress(0x10000)).is_none());

        assert!(regions
            .iter()
            .map(|x| (x.0, x.1))
            .eq(iterated_regions.iter().map(|x| *x)));

        let snapshot3 = snapshot2.memory_map();
        let res: GuestMemoryResult<()> = snapshot3.with_regions(|_, region| {
            assert_eq!(region.len(), region_size as GuestUsize);
            Ok(())
        });
        assert!(res.is_ok());
        let res: GuestMemoryResult<()> = snapshot3.with_regions_mut(|_, _| Ok(()));
        assert!(res.is_ok());
        assert_eq!(snapshot3.num_regions(), 2);
        assert!(snapshot3.find_region(GuestAddress(0x1000)).is_some());
        assert!(snapshot3.find_region(GuestAddress(0x10000)).is_none());
    }
}
