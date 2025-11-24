use libc::{c_void, fstat, off_t, pread, pwrite};
use std::fs::{self, File};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::{io, os::fd::IntoRawFd};

use crate::diskio::constants::{DIRECT_IO_ALIGNMENT, open_file_with_direct_io};

pub struct SharedFd {
    fd: RawFd,
    path: PathBuf,
    delete_on_drop: bool,
}

impl SharedFd {
    pub fn new_from_path(path: impl AsRef<Path>, delete_on_drop: bool) -> io::Result<Self> {
        let file = open_file_with_direct_io(path.as_ref())?;
        let fd = file.into_raw_fd();
        Ok(Self {
            fd,
            path: path.as_ref().to_path_buf(),
            delete_on_drop,
        })
    }

    /// Get the raw file descriptor
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd
    }
}

impl Drop for SharedFd {
    fn drop(&mut self) {
        // Close the file descriptor when SharedFd is dropped
        unsafe {
            libc::close(self.fd);
        }

        if self.delete_on_drop {
            if fs::remove_file(&self.path).is_ok() {
                sync_parent_directory(&self.path);
            }
        }
    }
}

#[cfg(target_family = "unix")]
fn sync_parent_directory(path: &Path) {
    if let Some(parent) = path.parent() {
        if let Ok(dir_file) = File::open(parent) {
            unsafe {
                #[cfg(target_os = "linux")]
                {
                    libc::syncfs(dir_file.as_raw_fd());
                }
                #[cfg(not(target_os = "linux"))]
                {
                    libc::fsync(dir_file.as_raw_fd());
                }
            }
            return;
        }
    }

    unsafe {
        libc::sync();
    }
}

#[cfg(not(target_family = "unix"))]
fn sync_parent_directory(_path: &Path) {}

/// Get the size of a file using its raw file descriptor
pub fn file_size_fd(fd: RawFd) -> io::Result<u64> {
    let mut stat_buf: libc::stat = unsafe { std::mem::zeroed() };

    let result = unsafe { fstat(fd, &mut stat_buf) };

    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(stat_buf.st_size as u64)
    }
}

/// Perform pread using raw file descriptor
///
/// This function reads data from a file at a specific offset without changing
/// the file position. It's thread-safe and doesn't require synchronization.
pub fn pread_fd(fd: RawFd, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let result = unsafe {
        pread(
            fd,
            buf.as_mut_ptr() as *mut c_void,
            buf.len(),
            offset as off_t,
        )
    };

    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result as usize)
    }
}

/// Perform pwrite using raw file descriptor
///
/// This function writes data to a file at a specific offset without changing
/// the file position. It's thread-safe and doesn't require synchronization.
pub fn pwrite_fd(fd: RawFd, buf: &[u8], offset: u64) -> io::Result<usize> {
    // Ensure buffer is aligned
    let buf_addr = buf.as_ptr() as usize;
    if buf_addr % DIRECT_IO_ALIGNMENT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Buffer is not properly aligned for Direct I/O",
        ));
    }

    // Ensure offset is aligned
    if offset % DIRECT_IO_ALIGNMENT as u64 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Offset is not properly aligned for Direct I/O",
        ));
    }

    // Ensure length is aligned
    if buf.len() % DIRECT_IO_ALIGNMENT != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Buffer length is not properly aligned for Direct I/O",
        ));
    }

    let result = unsafe {
        pwrite(
            fd,
            buf.as_ptr() as *const c_void,
            buf.len(),
            offset as off_t,
        )
    };

    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_file_deleted_on_drop() {
        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Keep the file by persisting it (so it's not deleted when NamedTempFile drops)
        let file = temp_file.persist(&path).unwrap();
        drop(file);

        // Verify file exists
        assert!(
            path.exists(),
            "File should exist before SharedFd is created"
        );

        // Create SharedFd with delete_on_drop = true
        {
            let _shared_fd = SharedFd::new_from_path(&path, true).unwrap();
            // SharedFd will be dropped at the end of this scope
        }

        // Verify file is deleted after drop
        assert!(
            !path.exists(),
            "File should be deleted after SharedFd is dropped"
        );
    }
}
