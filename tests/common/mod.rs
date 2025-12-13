#![allow(dead_code)]

use std::path::PathBuf;

pub mod sorter_behavior;

pub fn test_dir() -> PathBuf {
    test_dir_with_name("test_runs")
}

pub fn ovc_test_dir() -> PathBuf {
    test_dir_with_name("test_runs_ovc")
}

pub fn test_dir_with_name(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("./{}", name));
    std::fs::create_dir_all(&dir).expect("Failed to create test directory");
    dir
}
