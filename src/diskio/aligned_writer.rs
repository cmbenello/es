use std::io::{self, Write};
use std::sync::Arc;

use crate::diskio::aligned_buffer::AlignedBuffer;
use crate::diskio::constants::{DEFAULT_BUFFER_SIZE, DIRECT_IO_ALIGNMENT, align_up};
use crate::diskio::file::{SharedFd, pwrite_fd};
use crate::diskio::io_stats::IoStatsTracker;

/// AlignedWriter that uses GlobalFileManager for file operations
pub struct AlignedWriter {
    /// Raw file descriptor for direct operations
    fd: Arc<SharedFd>,
    /// Aligned buffer for writing
    buffer: AlignedBuffer,
    /// Current position in the file (aligned)
    file_offset: u64,
    /// Current position in the buffer
    buffer_pos: usize,
    /// Logical position in the file (unaligned)
    logical_pos: u64,
    /// I/O statistics tracker
    io_tracker: Option<IoStatsTracker>,
}

impl AlignedWriter {
    pub fn from_fd(fd: Arc<SharedFd>) -> io::Result<Self> {
        Ok(Self {
            fd,
            buffer: AlignedBuffer::new(DEFAULT_BUFFER_SIZE, DIRECT_IO_ALIGNMENT)?,
            file_offset: 0,
            buffer_pos: 0,
            logical_pos: 0,
            io_tracker: None,
        })
    }

    /// Create a new ManagedAlignedWriter with specified buffer size and I/O tracker
    pub fn from_fd_with_tracker(
        fd: Arc<SharedFd>,
        tracker: Option<IoStatsTracker>,
    ) -> io::Result<Self> {
        Ok(Self {
            fd,
            buffer: AlignedBuffer::new(DEFAULT_BUFFER_SIZE, DIRECT_IO_ALIGNMENT)?,
            file_offset: 0,
            buffer_pos: 0,
            logical_pos: 0,
            io_tracker: tracker,
        })
    }

    pub fn get_fd(&self) -> Arc<SharedFd> {
        self.fd.clone()
    }

    /// Flush the internal buffer to the file
    fn flush_buffer(&mut self) -> io::Result<()> {
        if self.buffer_pos == 0 {
            return Ok(());
        }

        // For Direct I/O, we need to write aligned chunks
        let write_len = align_up(self.buffer_pos, DIRECT_IO_ALIGNMENT);

        // Zero out the padding bytes
        for i in self.buffer_pos..write_len {
            self.buffer.as_mut_slice()[i] = 0;
        }

        // Write the aligned buffer using the raw file descriptor
        let written = pwrite_fd(
            self.fd.as_raw_fd(),
            &self.buffer.as_slice()[..write_len],
            self.file_offset,
        )?;

        if written != write_len {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "Failed to write all bytes",
            ));
        }

        // Update I/O statistics if tracker is present
        if let Some(ref tracker) = self.io_tracker {
            tracker.add_write(1, written as u64);
        }

        self.logical_pos += self.buffer_pos as u64;
        self.file_offset += write_len as u64;
        self.buffer_pos = 0;

        Ok(())
    }

    /// Get the current position in the file
    pub fn position(&self) -> u64 {
        self.logical_pos + self.buffer_pos as u64
    }

    /// Sync all data to disk
    pub fn sync(&mut self) -> io::Result<()> {
        self.flush()
    }
}

impl Write for AlignedWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut total_written = 0;

        while total_written < buf.len() {
            let available = DEFAULT_BUFFER_SIZE - self.buffer_pos;
            let to_copy = std::cmp::min(buf.len() - total_written, available);

            // Copy data to buffer
            self.buffer.as_mut_slice()[self.buffer_pos..self.buffer_pos + to_copy]
                .copy_from_slice(&buf[total_written..total_written + to_copy]);

            self.buffer_pos += to_copy;
            total_written += to_copy;

            // Flush if buffer is full
            if self.buffer_pos == DEFAULT_BUFFER_SIZE {
                self.flush_buffer()?;
            }
        }

        Ok(total_written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_buffer()
    }
}

impl Drop for AlignedWriter {
    fn drop(&mut self) {
        // Best effort flush on drop
        let _ = self.flush_buffer();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diskio::aligned_reader::AlignedReader;

    use std::fs;
    use std::io::Read;

    use std::path::Path;
    use tempfile::tempdir;

    fn create_test_file(dir: &Path, name: &str) -> io::Result<Arc<SharedFd>> {
        let file_path = dir.join(name);
        let fd = SharedFd::new_from_path(&file_path)?;
        Ok(Arc::new(fd))
    }

    #[test]
    fn test_basic_write() {
        let dir = tempdir().unwrap();
        let fd = create_test_file(dir.path(), "test.dat").unwrap();
        let mut writer = AlignedWriter::from_fd(fd).unwrap();
        let data = b"Hello, World!";
        writer.write_all(data).unwrap();
        writer.flush().unwrap();

        // Read back using regular reader to verify
        let content = fs::read(dir.path().join("test.dat")).unwrap();
        assert!(content.starts_with(b"Hello, World!"));
    }

    #[test]
    fn test_large_write() {
        let dir = tempdir().unwrap();
        let fd = create_test_file(dir.path(), "large_test.dat").unwrap();

        // Generate test data larger than buffer
        let test_data: Vec<u8> = (0..200_000).map(|i| (i % 256) as u8).collect();

        {
            let mut writer = AlignedWriter::from_fd(fd.clone()).unwrap();
            writer.write_all(&test_data).unwrap();
            writer.flush().unwrap();
        }

        // Verify using ManagedAlignedReader
        let mut reader = AlignedReader::from_fd(fd).unwrap();
        let mut read_data = vec![0u8; test_data.len()];
        reader.read_exact(&mut read_data).unwrap();

        assert_eq!(&read_data[..test_data.len()], &test_data[..]);
    }

    #[test]
    fn test_multiple_writers_different_files() {
        let dir = tempdir().unwrap();
        let fd1 = create_test_file(dir.path(), "file1.dat").unwrap();
        let fd2 = create_test_file(dir.path(), "file2.dat").unwrap();

        let mut writer1 = AlignedWriter::from_fd(fd1).unwrap();
        let mut writer2 = AlignedWriter::from_fd(fd2).unwrap();

        writer1.write_all(b"File 1 content").unwrap();
        writer2.write_all(b"File 2 content").unwrap();

        writer1.flush().unwrap();
        writer2.flush().unwrap();

        let content1 = fs::read(dir.path().join("file1.dat")).unwrap();
        let content2 = fs::read(dir.path().join("file2.dat")).unwrap();

        assert!(content1.starts_with(b"File 1 content"));
        assert!(content2.starts_with(b"File 2 content"));
    }

    #[test]
    fn test_position_tracking() {
        let dir = tempdir().unwrap();
        let fd = create_test_file(dir.path(), "position_test.dat").unwrap();

        let mut writer = AlignedWriter::from_fd(fd).unwrap();

        assert_eq!(writer.position(), 0);

        writer.write_all(b"12345").unwrap();
        assert_eq!(writer.position(), 5);

        writer.write_all(b"67890").unwrap();
        assert_eq!(writer.position(), 10);

        writer.flush().unwrap();
        assert_eq!(writer.position(), 10);
    }

    #[test]
    fn test_write_patterns() {
        let dir = tempdir().unwrap();
        let fd = create_test_file(dir.path(), "test.dat").unwrap();

        {
            let mut writer = AlignedWriter::from_fd(fd.clone()).unwrap();

            // Write pattern: small, large, small
            writer.write_all(b"small").unwrap();

            let large = vec![b'X'; 100_000];
            writer.write_all(&large).unwrap();

            writer.write_all(b"end").unwrap();
            writer.flush().unwrap();
        }

        // Verify pattern
        let mut reader = AlignedReader::from_fd(fd).unwrap();
        let mut buf = vec![0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"small");

        let mut buf = vec![0u8; 100_000];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, vec![b'X'; 100_000]);

        let mut buf = vec![0u8; 3];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"end");
    }

    #[test]
    fn test_auto_flush_on_drop() {
        let dir = tempdir().unwrap();
        let fd = create_test_file(dir.path(), "auto_flush_test.dat").unwrap();

        {
            let mut writer = AlignedWriter::from_fd(fd).unwrap();
            writer.write_all(b"Auto flush test").unwrap();
            // No explicit flush - should flush on drop
        }

        // Verify data was written
        let content = fs::read(dir.path().join("auto_flush_test.dat")).unwrap();
        assert!(content.starts_with(b"Auto flush test"));
    }

    #[test]
    fn test_concurrent_write_read() {
        let dir = tempdir().unwrap();
        let fd = create_test_file(dir.path(), "test.dat").unwrap();

        // First write some initial data
        {
            let mut writer = AlignedWriter::from_fd(fd.clone()).unwrap();
            let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
            writer.write_all(&data).unwrap();
            writer.flush().unwrap();
        }

        // Now create a reader and writer using same file manager
        let mut reader = AlignedReader::from_fd(fd).unwrap();

        // Read first part
        let mut buf = vec![0u8; 1000];
        reader.read_exact(&mut buf).unwrap();

        // Verify we read the correct data
        let expected: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
        assert_eq!(buf, expected);
    }

    #[test]
    fn test_io_tracker() {
        let dir = tempdir().unwrap();
        let fd = create_test_file(dir.path(), "test.dat").unwrap();

        let tracker = IoStatsTracker::new();

        let mut writer = AlignedWriter::from_fd_with_tracker(fd, Some(tracker.clone())).unwrap();

        // Write some data
        let data = vec![b'X'; 10_000];
        writer.write_all(&data).unwrap();

        // Check that nothing is tracked yet (not flushed)
        let (ops, bytes) = tracker.get_write_stats();
        assert_eq!(ops, 0);
        assert_eq!(bytes, 0);

        // Flush and check stats
        writer.flush().unwrap();
        let (ops, bytes) = tracker.get_write_stats();
        assert!(ops > 0);
        assert_eq!(bytes as usize, 10_000);

        // Write more data
        writer.write_all(&data).unwrap();
        writer.flush().unwrap();

        // Check updated stats
        let (ops2, bytes2) = tracker.get_write_stats();
        assert!(ops2 > ops);
        assert_eq!(bytes2 as usize, 20_000);
    }
}
