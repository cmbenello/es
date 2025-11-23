#![allow(dead_code)]

use es::{InMemInput, Sorter};
use rand::seq::SliceRandom;
use std::sync::Arc;
use std::thread;

pub fn basic_sort<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let data = vec![
        (b"z".to_vec(), b"26".to_vec()),
        (b"a".to_vec(), b"1".to_vec()),
        (b"m".to_vec(), b"13".to_vec()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    assert_eq!(
        results,
        vec![
            (b"a".to_vec(), b"1".to_vec()),
            (b"m".to_vec(), b"13".to_vec()),
            (b"z".to_vec(), b"26".to_vec()),
        ]
    );
}

pub fn load_sort_store<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in 0..500 {
        let key = format!("{:06}", 500 - i).into_bytes();
        let value = format!("value_{:06}", i).into_bytes();
        data.push((key, value));
    }
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

pub fn empty_input<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let input = InMemInput { data: vec![] };
    let output = sorter.sort(Box::new(input)).unwrap();
    assert!(output.iter().next().is_none());
}

pub fn single_element<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let input = InMemInput {
        data: vec![(b"only".to_vec(), b"one".to_vec())],
    };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    assert_eq!(results, vec![(b"only".to_vec(), b"one".to_vec())]);
}

pub fn large_dataset_with_threads<S, F>(mut factory: F)
where
    S: Sorter,
    F: FnMut(usize) -> S,
{
    for num_threads in [1usize, 4usize] {
        let mut sorter = factory(num_threads);
        let mut data = Vec::new();
        for i in 0..100_000 {
            let key = format!("{:08}", (i * 7919) % 100_000).into_bytes();
            let value = format!("value_{}", i).into_bytes();
            data.push((key, value));
        }

        let input = InMemInput { data };
        let output = sorter.sort(Box::new(input)).unwrap();
        let results: Vec<_> = output.iter().collect();
        for i in 0..100_000 {
            assert_eq!(results[i].0, format!("{:08}", i).as_bytes());
        }
    }
}

pub fn small_buffer_external_sort<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in 0..5000 {
        let key = format!("{:05}", 5000 - i - 1).into_bytes();
        data.push((key, vec![b'v'; 20]));
    }

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    for i in 0..5000 {
        assert_eq!(results[i].0, format!("{:05}", i).as_bytes());
    }
}

pub fn duplicate_keys<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in 0..100 {
        for j in 0..10 {
            let key = format!("key_{:03}", i).into_bytes();
            let value = format!("value_{}_{}", i, j).into_bytes();
            data.push((key, value));
        }
    }
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    for key in 0..100 {
        let expected = format!("key_{:03}", key).into_bytes();
        let count = results.iter().filter(|(k, _)| k == &expected).count();
        assert_eq!(count, 10);
    }
}

pub fn run_generation_stats_small_buffer<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in (0..10).rev() {
        data.push((format!("{:02}", i).into_bytes(), vec![i as u8]));
    }
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let stats = output.stats();
    let run_lengths: Vec<_> = stats
        .run_gen_stats
        .runs_info
        .iter()
        .map(|info| info.entries)
        .collect();
    assert_eq!(run_lengths, vec![2, 2, 2, 2, 2]);
}

pub fn variable_sized_data<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in 0..1000 {
        let key = match i % 3 {
            0 => format!("{:03}", i).into_bytes(),
            1 => format!("longer_key_{:06}", i).into_bytes(),
            _ => format!("very_long_key_with_more_data_{:09}", i).into_bytes(),
        };
        let value = vec![b'x'; (i % 100) + 1];
        data.push((key, value));
    }
    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

pub fn binary_data_with_nulls<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in 0..256 {
        let key = vec![i as u8, 0, 255 - i as u8];
        let value = vec![0, i as u8, 0, 255 - i as u8];
        data.push((key, value));
    }
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    for i in 1..results.len() {
        assert!(results[i - 1].0 <= results[i].0);
    }
}

pub fn buffer_boundary_cases<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in 0..100 {
        data.push((format!("k{:02}", i).into_bytes(), b"v".to_vec()));
    }
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    for i in 0..100 {
        assert_eq!(results[i].0, format!("k{:02}", i).as_bytes());
    }
}

pub fn entry_too_large_errors<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let data = vec![(vec![b'k'; 100], vec![b'v'; 100])];
    let input = InMemInput { data };
    assert!(sorter.sort(Box::new(input)).is_err());
}

pub fn concurrent_sorters<S, F>(factory: F)
where
    S: Sorter + Send + 'static,
    F: Fn() -> S + Send + Sync + 'static,
{
    let factory = Arc::new(factory);
    let handles: Vec<_> = (0..4)
        .map(|id| {
            let factory = factory.clone();
            thread::spawn(move || {
                let mut sorter = factory();
                let mut data = Vec::new();
                for i in 0..1000 {
                    let key = format!("sorter_{}_key_{:04}", id, i).into_bytes();
                    let value = format!("value_{}", i).into_bytes();
                    data.push((key, value));
                }
                let input = InMemInput { data };
                let output = sorter.sort(Box::new(input)).unwrap();
                let results: Vec<_> = output.iter().collect();
                for i in 1..results.len() {
                    assert!(results[i - 1].0 <= results[i].0);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

pub fn large_values<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let large_value = vec![b'x'; 500];
    let data = vec![
        (b"c".to_vec(), large_value.clone()),
        (b"a".to_vec(), large_value.clone()),
        (b"b".to_vec(), large_value.clone()),
    ];
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[1].0, b"b");
    assert_eq!(results[2].0, b"c");
    assert_eq!(results[0].1.len(), 500);
}

pub fn stats_populated<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let mut data = Vec::new();
    for i in (0..5000).rev() {
        data.push((
            format!("key_{:02}", i).into_bytes(),
            format!("value_{}", i).into_bytes(),
        ));
    }
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let stats = output.stats();
    assert!(stats.run_gen_stats.num_runs > 0);
    if stats.run_gen_stats.num_runs > 1 {
        let total_merge: u128 = stats.per_merge_stats.iter().map(|m| m.time_ms).sum();
        assert!(total_merge > 0);
    }
}

pub fn binary_keys<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let data = vec![
        (vec![255, 255, 255], b"max".to_vec()),
        (vec![0, 0, 0], b"min".to_vec()),
        (vec![127, 127, 127], b"mid".to_vec()),
        (vec![1, 2, 3], b"low".to_vec()),
        (vec![200, 201, 202], b"high".to_vec()),
    ];
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    assert_eq!(results[0].0, vec![0, 0, 0]);
    assert_eq!(results[4].0, vec![255, 255, 255]);
}

pub fn variable_length_keys<S, F>(factory: F)
where
    S: Sorter,
    F: FnOnce() -> S,
{
    let mut sorter = factory();
    let data = vec![
        (b"zzz".to_vec(), b"3_chars".to_vec()),
        (b"a".to_vec(), b"1_char".to_vec()),
        (b"bb".to_vec(), b"2_chars".to_vec()),
        (b"aaaa".to_vec(), b"4_chars".to_vec()),
        (b"aaa".to_vec(), b"3_chars_a".to_vec()),
    ];
    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();
    let results: Vec<_> = output.iter().collect();
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[4].0, b"zzz");
}
