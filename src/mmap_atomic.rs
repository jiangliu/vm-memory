// Copyright (C) 2019 Alibaba Cloud Computing. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A default implementation of the GuestMemory trait by mmap()-ing guest's memory into the current
//! process.
extern crate arc_swap;

use crate::guest_memory::FileOffset;
use crate::{Error, GuestAddress, GuestMemory, GuestMemoryGuard, GuestMemoryMmap, GuestRegionMmap};
use arc_swap::{ArcSwap, Guard};
use std::borrow::Borrow;
use std::result;
use std::sync::Arc;

#[derive(Debug)]
enum Either {
    Object(Arc<ArcSwap<GuestMemoryMmap>>),
    Guard(Guard<'static, Arc<GuestMemoryMmap>>),
}

/// Tracks memory regions allocated/mapped for the guest in the current process.
///
/// This implementation uses ArcSwap to provide RCU-like lock strategy to support memory hotplug.
/// The implementation is really a little over complex, hope that it deserves the complexity in
/// case of performance.
#[derive(Debug)]
pub struct GuestMemoryMmapAtomic {
    inner: Either,
}

impl GuestMemoryMmapAtomic {
    /// Creates a container and allocates anonymous memory for guest memory regions.
    /// Valid memory regions are specified as a slice of (Address, Size) tuples sorted by Address.
    pub fn new(ranges: &[(GuestAddress, usize)]) -> result::Result<Self, Error> {
        let mmap = GuestMemoryMmap::with_files(ranges.iter().map(|r| (r.0, r.1, None)))?;
        Ok(GuestMemoryMmapAtomic {
            inner: Either::Object(Arc::new(ArcSwap::new(Arc::new(mmap)))),
        })
    }

    /// Creates a container and allocates anonymous memory for guest memory regions.
    /// Valid memory regions are specified as a sequence of (Address, Size, Option<FileOffset>)
    /// tuples sorted by Address.
    pub fn with_files<A, T>(ranges: T) -> result::Result<Self, Error>
    where
        A: Borrow<(GuestAddress, usize, Option<FileOffset>)>,
        T: IntoIterator<Item = A>,
    {
        let mmap = GuestMemoryMmap::with_files(ranges)?;
        Ok(GuestMemoryMmapAtomic {
            inner: Either::Object(Arc::new(ArcSwap::new(Arc::new(mmap)))),
        })
    }

    /// Creates a new `GuestMemoryMmap` from a vector of regions.
    ///
    /// # Arguments
    ///
    /// * `regions` - The vector of regions.
    ///               The regions shouldn't overlap and they should be sorted
    ///               by the starting address.
    pub fn from_regions(mut regions: Vec<GuestRegionMmap>) -> result::Result<Self, Error> {
        Self::from_arc_regions(regions.drain(..).map(Arc::new).collect())
    }

    /// Creates a new `GuestMemoryMmap` from a vector of Arc regions.
    ///
    /// Similar to the constructor from_regions() as it returns a
    /// GuestMemoryMmap. The need for this constructor is to provide a way for
    /// consumer of this API to create a new GuestMemoryMmap based on existing
    /// regions coming from an existing GuestMemoryMmap instance.
    ///
    /// # Arguments
    ///
    /// * `regions` - The vector of Arc regions.
    ///               The regions shouldn't overlap and they should be sorted
    ///               by the starting address.
    pub fn from_arc_regions(regions: Vec<Arc<GuestRegionMmap>>) -> result::Result<Self, Error> {
        let mmap = GuestMemoryMmap::from_arc_regions(regions)?;
        Ok(GuestMemoryMmapAtomic {
            inner: Either::Object(Arc::new(ArcSwap::new(Arc::new(mmap)))),
        })
    }
}

impl GuestMemory for GuestMemoryMmapAtomic {
    type R = GuestRegionMmap;

    fn snapshot(&self) -> GuestMemoryGuard<'_, Self>
    where
        Self: std::marker::Sized,
    {
        let inner = match self.inner {
            Either::Object(ref obj) => Either::Guard(obj.load()),
            Either::Guard(ref _guard) => panic!("can't create guard from guard object!!!"),
        };
        GuestMemoryGuard::Valued(GuestMemoryMmapAtomic { inner })
    }

    fn num_regions(&self) -> usize {
        panic!("please use GuestMemoryMmapAtomic::snapshot().num_regions()");
    }

    fn find_region(&self, _addr: GuestAddress) -> Option<&Self::R> {
        panic!("please use GuestMemoryMmapAtomic::snapshot().find_region(addr)");
    }

    fn with_regions<F, E>(&self, cb: F) -> result::Result<(), E>
    where
        F: Fn(usize, &Self::R) -> result::Result<(), E>,
    {
        match self.inner {
            Either::Object(ref o) => o.load().with_regions(cb),
            Either::Guard(ref g) => g.with_regions(cb),
        }
    }

    fn with_regions_mut<F, E>(&self, cb: F) -> result::Result<(), E>
    where
        F: FnMut(usize, &Self::R) -> result::Result<(), E>,
    {
        match self.inner {
            Either::Object(ref o) => o.load().with_regions_mut(cb),
            Either::Guard(ref g) => g.with_regions_mut(cb),
        }
    }

    fn map_and_fold<F, G, T>(&self, init: T, mapf: F, foldf: G) -> T
    where
        F: Fn((usize, &Self::R)) -> T,
        G: Fn(T, T) -> T,
    {
        match self.inner {
            Either::Object(ref o) => o.load().map_and_fold(init, mapf, foldf),
            Either::Guard(ref g) => g.map_and_fold(init, mapf, foldf),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{guest_memory, GuestMemoryRegion, GuestUsize};

    #[test]
    fn test_snapshot() {
        let region_size = 0x400;
        let regions = vec![
            (GuestAddress(0x0), region_size),
            (GuestAddress(0x1000), region_size),
        ];
        let mut iterated_regions = Vec::new();
        let gm = GuestMemoryMmapAtomic::new(&regions).unwrap();
        let snapshot = gm.snapshot();

        let res: guest_memory::Result<()> = snapshot.with_regions(|_, region| {
            assert_eq!(region.len(), region_size as GuestUsize);
            Ok(())
        });
        assert!(res.is_ok());
        let res: guest_memory::Result<()> = snapshot.with_regions_mut(|_, region| {
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

    #[should_panic]
    #[test]
    fn test_num_regions_panic() {
        let region_size = 0x400;
        let regions = vec![
            (GuestAddress(0x0), region_size),
            (GuestAddress(0x1000), region_size),
        ];
        let gm = GuestMemoryMmapAtomic::new(&regions).unwrap();
        let _ = gm.num_regions();
    }

    #[should_panic]
    #[test]
    fn test_find_region_panic() {
        let region_size = 0x400;
        let regions = vec![
            (GuestAddress(0x0), region_size),
            (GuestAddress(0x1000), region_size),
        ];
        let gm = GuestMemoryMmapAtomic::new(&regions).unwrap();
        let _ = gm.find_region(GuestAddress(0x1001));
    }
}
