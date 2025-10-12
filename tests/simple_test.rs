// Simple smoke tests that run quickly - for basic validation
mod common;
use common::test_dir;

use es::{ExternalSorter, InMemInput, Sorter};

#[test]
fn test_basic_sort() {
    let mut sorter = ExternalSorter::new(2, 512, 2, 10000, test_dir());

    let data = vec![
        (b"c".to_vec(), b"3".to_vec()),
        (b"a".to_vec(), b"1".to_vec()),
        (b"b".to_vec(), b"2".to_vec()),
    ];

    let input = InMemInput { data };
    let output = sorter.sort(Box::new(input)).unwrap();

    let results: Vec<_> = output.iter().collect();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, b"a");
    assert_eq!(results[1].0, b"b");
    assert_eq!(results[2].0, b"c");
}
