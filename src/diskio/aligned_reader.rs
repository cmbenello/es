use std::io::{self, Read, Seek, SeekFrom};
use std::sync::Arc;

use bytes::Bytes;
use parquet::errors::Result as ParquetResult;
use parquet::file::reader::{ChunkReader, Length};

use crate::diskio::aligned_buffer::AlignedBuffer;
use crate::diskio::constants::{
    DEFAULT_BUFFER_SIZE, DIRECT_IO_ALIGNMENT, align_down, align_up, offset_within_block,
};
use crate::diskio::file::file_size_fd;
use crate::diskio::file::{SharedFd, pread_fd};
use crate::diskio::io_stats::IoStatsTracker;

/// AlignedReader that uses GlobalFileManager for file operations
pub struct AlignedReader {
    /// Raw file descriptor for direct operations
    fd: Arc<SharedFd>,
    /// Aligned buffer for reading
    buffer: AlignedBuffer,
    /// Current position in the file (aligned)
    file_offset: u64,
    /// Current position in the buffer
    buffer_offset: usize,
    /// Number of valid bytes in the buffer
    buffer_valid_len: usize,
    /// I/O statistics tracker
    io_tracker: Option<IoStatsTracker>,
    /// Logical position in the file (unaligned)
    logical_pos: u64,
    /// File size
    file_size: u64,
}

impl AlignedReader {
    pub fn from_fd(fd: Arc<SharedFd>) -> io::Result<Self> {
        Self::from_fd_with_tracer(fd, None)
    }

    pub fn from_fd_with_tracer(
        fd: Arc<SharedFd>,
        tracker: Option<IoStatsTracker>,
    ) -> io::Result<Self> {
        Self::from_fd_with_start_position(fd, 0, tracker)
    }

    pub fn from_fd_with_start_position(
        fd: Arc<SharedFd>,
        start_byte: u64,
        tracker: Option<IoStatsTracker>,
    ) -> io::Result<Self> {
        // Calculate aligned position and skip bytes
        let file_size = file_size_fd(fd.as_raw_fd()).map_err(io::Error::other)?;
        let aligned_pos = align_down(start_byte, DIRECT_IO_ALIGNMENT as u64);
        let skip_bytes = offset_within_block(start_byte, DIRECT_IO_ALIGNMENT as u64);

        let mut reader = Self {
            fd,
            buffer: AlignedBuffer::new(DEFAULT_BUFFER_SIZE, DIRECT_IO_ALIGNMENT)?,
            file_offset: aligned_pos,
            buffer_offset: 0,
            buffer_valid_len: 0,
            io_tracker: tracker,
            logical_pos: start_byte,
            file_size,
        };

        // If we need to skip bytes, fill the buffer and set buffer_offset
        if skip_bytes > 0 {
            reader.refill_buffer()?;
            reader.buffer_offset = skip_bytes.min(reader.buffer_valid_len);
        }

        Ok(reader)
    }

    /// Refill the internal buffer from the file
    /// If the result is 0, self.buffer_valid_len and self.buffer_offset are kept unchanged
    /// Otherwise, they are updated to reflect the new state
    fn refill_buffer(&mut self) -> io::Result<usize> {
        // Read aligned data from file using the raw file descriptor
        let bytes_read = pread_fd(
            self.fd.as_raw_fd(),
            self.buffer.as_mut_slice(),
            self.file_offset,
        )?;

        // Update I/O statistics if tracker is present
        if let Some(ref tracker) = self.io_tracker {
            tracker.add_read(1, bytes_read as u64);
        }

        if bytes_read == 0 {
            // EOF reached - for now just return 0
            // TODO: We might want to track file sizes separately if needed
            Ok(0)
        } else {
            self.buffer_valid_len = bytes_read;
            self.buffer_offset = 0;

            // Handle case where we read less than buffer size (near EOF)
            if bytes_read < DEFAULT_BUFFER_SIZE {
                // We've reached EOF - just round up to next alignment boundary
                let next_aligned = align_up(
                    (self.file_offset + bytes_read as u64) as usize,
                    DEFAULT_BUFFER_SIZE,
                ) as u64;
                self.file_offset = next_aligned;
            } else {
                // Normal case - advance by buffer size
                self.file_offset += DEFAULT_BUFFER_SIZE as u64;
            }

            Ok(bytes_read)
        }
    }

    /// Seek to a specific position in the file
    pub fn seek(&mut self, pos: u64) -> io::Result<()> {
        // Clamp position to file size
        let clamped_pos = pos.min(self.file_size);

        // Update logical position
        self.logical_pos = clamped_pos;

        // For Direct I/O, we need to align the file offset
        let aligned_pos = align_down(clamped_pos, DEFAULT_BUFFER_SIZE as u64);
        let offset_in_block = offset_within_block(clamped_pos, DEFAULT_BUFFER_SIZE as u64);

        // If seeking within the current buffer, just adjust offset
        if self.buffer_valid_len > 0 && self.file_offset >= DEFAULT_BUFFER_SIZE as u64 {
            let current_buffer_start = self.file_offset - DEFAULT_BUFFER_SIZE as u64;
            let current_buffer_end = current_buffer_start + self.buffer_valid_len as u64;

            if clamped_pos >= current_buffer_start && clamped_pos < current_buffer_end {
                // We can seek within the current buffer
                let new_offset = (clamped_pos - current_buffer_start) as usize;
                self.buffer_offset = new_offset;
                return Ok(());
            }
        }

        // Set the new aligned position
        self.file_offset = aligned_pos;
        self.buffer_offset = 0;
        self.buffer_valid_len = 0; // Force refill on next read

        // If we need to skip some bytes after alignment, we'll do it on next read
        if offset_in_block > 0 && clamped_pos < self.file_size {
            // Read the buffer to position at the right offset
            self.refill_buffer()?;
            self.buffer_offset = offset_in_block.min(self.buffer_valid_len);
        }

        Ok(())
    }

    /// Get the current position in the file
    pub fn position(&self) -> u64 {
        self.logical_pos
    }

    /// Skip to the next newline character and return the number of bytes skipped
    pub fn skip_to_newline(&mut self) -> Result<usize, String> {
        let start_pos = self.position();

        loop {
            // Make sure we have data in the buffer
            if self.buffer_offset >= self.buffer_valid_len {
                let read = self
                    .refill_buffer()
                    .map_err(|e| format!("Failed to refill buffer: {}", e))?;
                if read == 0 {
                    // EOF reached - return actual bytes skipped based on position
                    let end_pos = self.position();
                    return Ok((end_pos - start_pos) as usize);
                }
            }

            // Check remaining buffer for newline
            if let Some(pos) = self.buffer.as_slice()[self.buffer_offset..self.buffer_valid_len]
                .iter()
                .position(|&b| b == b'\n')
            {
                let bytes_skipped = pos + 1;
                self.buffer_offset += bytes_skipped;
                self.logical_pos += bytes_skipped as u64;
                return Ok((self.logical_pos - start_pos) as usize);
            }

            // No newline found in current buffer, skip to end
            let bytes_in_buffer = self.buffer_valid_len - self.buffer_offset;
            self.logical_pos += bytes_in_buffer as u64;
            self.buffer_offset = self.buffer_valid_len;
        }
    }

    /// Read a line into the provided string buffer
    /// Returns the number of bytes read, including the newline character if present
    pub fn read_line(&mut self, buf: &mut String) -> io::Result<usize> {
        let start_len = buf.len();

        loop {
            // Make sure we have data in the buffer
            if self.buffer_offset >= self.buffer_valid_len {
                let read = self.refill_buffer()?;
                if read == 0 {
                    // EOF reached
                    return Ok(buf.len() - start_len);
                }
            }

            // Look for newline in the remaining buffer
            let search_start = self.buffer_offset;
            let search_end = self.buffer_valid_len;
            let buffer_slice = &self.buffer.as_slice()[search_start..search_end];

            if let Some(newline_pos) = buffer_slice.iter().position(|&b| b == b'\n') {
                // Found newline - read up to and including it
                let line_bytes = &buffer_slice[..=newline_pos];

                // Convert to string and append
                match std::str::from_utf8(line_bytes) {
                    Ok(s) => buf.push_str(s),
                    Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
                }

                self.buffer_offset = search_start + newline_pos + 1;
                self.logical_pos += (newline_pos + 1) as u64;
                return Ok(buf.len() - start_len);
            } else {
                // No newline found - read entire remaining buffer
                match std::str::from_utf8(buffer_slice) {
                    Ok(s) => buf.push_str(s),
                    Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
                }

                self.buffer_offset = search_end;
                self.logical_pos += buffer_slice.len() as u64;
                // Continue to next buffer
            }
        }
    }
}

impl Read for AlignedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // If we're at or past EOF, return 0
        if self.logical_pos >= self.file_size {
            return Ok(0);
        }

        let mut total_read = 0;

        loop {
            // If buffer is empty, refill it
            if self.buffer_offset >= self.buffer_valid_len {
                let read = self.refill_buffer().unwrap();
                if read == 0 {
                    // EOF reached
                    break;
                }
            }

            // Copy data from buffer to output
            let available = self.buffer_valid_len - self.buffer_offset;
            let to_copy = std::cmp::min(buf.len() - total_read, available);

            buf[total_read..total_read + to_copy].copy_from_slice(
                &self.buffer.as_slice()[self.buffer_offset..self.buffer_offset + to_copy],
            );

            self.buffer_offset += to_copy;
            self.logical_pos += to_copy as u64;
            total_read += to_copy;

            if total_read == buf.len() {
                break;
            }
        }

        Ok(total_read)
    }
}

impl Seek for AlignedReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(pos) => pos,
            SeekFrom::End(offset) => {
                if offset >= 0 {
                    self.file_size.saturating_add(offset as u64)
                } else {
                    self.file_size.saturating_sub((-offset) as u64)
                }
            }
            SeekFrom::Current(offset) => {
                let current = self.position();
                if offset >= 0 {
                    current.saturating_add(offset as u64)
                } else {
                    current.saturating_sub((-offset) as u64)
                }
            }
        };

        // Use our existing seek method
        self.seek(new_pos)?;
        Ok(self.logical_pos)
    }

    fn stream_position(&mut self) -> io::Result<u64> {
        Ok(self.logical_pos)
    }
}

/// A wrapper around a file path that implements ChunkReader for Direct I/O using GlobalFileManager
pub struct AlignedChunkReader {
    fd: Arc<SharedFd>,
    file_size: u64,
    io_tracker: Option<IoStatsTracker>,
}

impl AlignedChunkReader {
    pub fn new(fd: Arc<SharedFd>) -> Result<Self, String> {
        Self::new_with_tracker(fd, None)
    }

    pub fn new_with_tracker(
        fd: Arc<SharedFd>,
        tracker: Option<IoStatsTracker>,
    ) -> Result<Self, String> {
        let file_size =
            file_size_fd(fd.as_raw_fd()).map_err(|e| format!("Failed to get file size: {}", e))?;

        Ok(Self {
            fd: fd.clone(),
            file_size,
            io_tracker: tracker,
        })
    }
}

impl Length for AlignedChunkReader {
    fn len(&self) -> u64 {
        self.file_size
    }
}

impl ChunkReader for AlignedChunkReader {
    type T = AlignedReader;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        // Create a new reader starting at the specified position
        let reader = AlignedReader::from_fd_with_start_position(
            self.fd.clone(),
            start,
            self.io_tracker.clone(),
        )
        .map_err(|e| parquet::errors::ParquetError::General(e.to_string()))?;

        Ok(reader)
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        let mut reader = self.get_read(start)?;
        let mut buffer = vec![0u8; length];
        reader.read_exact(&mut buffer).map_err(|e| {
            parquet::errors::ParquetError::General(format!("Failed to read: {}", e))
        })?;
        Ok(Bytes::from(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use tempfile::TempDir;

    fn create_test_file(
        dir: &Path,
        name: &str,
        size: usize,
    ) -> io::Result<(Arc<SharedFd>, Vec<u8>)> {
        let path = dir.join(name);
        let mut data = vec![0u8; size];
        for i in 0..size {
            data[i] = (i % 256) as u8;
        }

        // Write directly to file
        let mut file = File::create(&path)?;
        file.write_all(&data)?;
        file.sync_all()?;
        drop(file); // Close the write file

        let fd = Arc::new(SharedFd::new_from_path(path, true)?);
        Ok((fd, data))
    }

    fn create_test_file_with_data(
        dir: &Path,
        name: &str,
        data: &[u8],
    ) -> io::Result<Arc<SharedFd>> {
        let path = dir.join(name);
        let mut file = File::create(&path)?;
        file.write_all(data)?;
        file.sync_all()?;
        drop(file);

        let fd = Arc::new(SharedFd::new_from_path(path, true)?);
        Ok(fd)
    }

    #[test]
    fn test_basic_read() {
        let temp_dir = TempDir::new().unwrap();
        let test_size = 8192;
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", test_size).unwrap();

        let mut reader = AlignedReader::from_fd(file).unwrap();

        let mut buf = vec![0u8; test_size];
        let bytes_read = reader.read(&mut buf).unwrap();

        assert_eq!(bytes_read, test_size);
        assert_eq!(buf, test_data);
    }

    #[test]
    fn test_small_reads() {
        let temp_dir = TempDir::new().unwrap();
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", 8192).unwrap();

        let mut content = vec![0u8; 8192];
        let read = pread_fd(file.as_raw_fd(), &mut content, 0).unwrap();
        assert_eq!(read, 8192);

        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Read in small chunks
        let chunk_size = 100;
        let mut all_data = Vec::new();
        let mut buf = vec![0u8; chunk_size];

        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => all_data.extend_from_slice(&buf[..n]),
                Err(e) => panic!("Read error: {}", e),
            }
        }

        assert_eq!(all_data, test_data);
    }

    #[test]
    fn test_seek_operations() {
        let temp_dir = TempDir::new().unwrap();
        let test_size = 8192;
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", test_size).unwrap();

        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Seek to middle
        reader.seek(4096).unwrap();
        let mut buf = vec![0u8; 100];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, &test_data[4096..4196]);

        // Seek to beginning
        reader.seek(0).unwrap();
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, &test_data[0..100]);

        // Seek to near end
        reader.seek((test_size - 50) as u64).unwrap();
        let mut buf = vec![0u8; 50];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 50);
        assert_eq!(&buf[..n], &test_data[test_size - 50..]);
    }

    #[test]
    fn test_position_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let (file, _) = create_test_file(temp_dir.path(), "test.dat", 8192).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        assert_eq!(reader.position(), 0);

        // Read some data
        let mut buf = vec![0u8; 1000];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(reader.position(), 1000);

        // Seek and check position
        reader.seek(2000).unwrap();
        assert_eq!(reader.position(), 2000);

        // Read more and check
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(reader.position(), 3000);
    }

    #[test]
    fn test_multiple_readers_same_file() {
        let temp_dir = TempDir::new().unwrap();
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", 8192).unwrap();

        // Create two readers for the same file
        let mut reader1 = AlignedReader::from_fd(file.clone()).unwrap();
        let mut reader2 = AlignedReader::from_fd(file).unwrap();

        // Read from different positions
        reader1.seek(0).unwrap();
        reader2.seek(4096).unwrap();

        let mut buf1 = vec![0u8; 100];
        let mut buf2 = vec![0u8; 100];

        reader1.read_exact(&mut buf1).unwrap();
        reader2.read_exact(&mut buf2).unwrap();

        assert_eq!(buf1, &test_data[0..100]);
        assert_eq!(buf2, &test_data[4096..4196]);
    }

    #[test]
    fn test_eof_handling() {
        let temp_dir = TempDir::new().unwrap();
        let (file, _) = create_test_file(temp_dir.path(), "test.dat", 1024).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Seek near end
        reader.seek(1000).unwrap();

        // Try to read more than available
        let mut buf = vec![0u8; 100];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 24); // Should only read remaining bytes

        // Next read should return 0 (EOF)
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_seek_to_non_aligned_offsets() {
        let temp_dir = TempDir::new().unwrap();
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", 8192).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Test various non-aligned offsets
        let test_offsets = vec![
            1,    // Just past start
            17,   // Random small offset
            511,  // One byte before alignment
            513,  // One byte after alignment
            1023, // One byte before next alignment
            1337, // Random offset
            4095, // One byte before page boundary
            4097, // One byte after page boundary
        ];

        for offset in test_offsets {
            reader.seek(offset).unwrap();
            assert_eq!(reader.position(), offset);

            // Read and verify data at this position
            let mut buf = vec![0u8; 10];
            let n = reader.read(&mut buf).unwrap();
            assert!(n > 0);
            assert_eq!(&buf[..n], &test_data[offset as usize..offset as usize + n]);
        }
    }

    #[test]
    fn test_seek_non_aligned_read_across_boundaries() {
        let temp_dir = TempDir::new().unwrap();
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", 8192).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Seek to non-aligned position just before an alignment boundary
        reader.seek(510).unwrap();

        // Read across the alignment boundary
        let mut buf = vec![0u8; 100];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, &test_data[510..610]);

        // Seek to another non-aligned position
        reader.seek(4090).unwrap();

        // Read across multiple alignment boundaries
        let mut buf = vec![0u8; 1000];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, &test_data[4090..5090]);
    }

    #[test]
    fn test_multiple_seeks_and_reads() {
        let temp_dir = TempDir::new().unwrap();
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", 8192).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Perform multiple seeks and reads
        let operations = vec![
            (100, 50),   // Seek to 100, read 50 bytes
            (2000, 100), // Seek to 2000, read 100 bytes
            (50, 25),    // Seek backwards to 50, read 25 bytes
            (7000, 200), // Seek near end, read 200 bytes
            (513, 511),  // Non-aligned to non-aligned read
        ];

        for (seek_pos, read_len) in operations {
            reader.seek(seek_pos).unwrap();
            let mut buf = vec![0u8; read_len];
            reader.read_exact(&mut buf).unwrap();
            assert_eq!(
                buf,
                &test_data[seek_pos as usize..(seek_pos as usize + read_len)]
            );
        }
    }

    #[test]
    fn test_skip_to_newline() {
        let temp_dir = TempDir::new().unwrap();

        // Create test data with newlines
        let test_data = b"First line\nSecond line\nThird line\n";
        let file = create_test_file_with_data(temp_dir.path(), "test.dat", test_data).unwrap();

        // Test skipping from middle of first line
        let mut reader = AlignedReader::from_fd(file).unwrap();
        reader.seek(5).unwrap(); // Position at " line"

        let skipped = reader.skip_to_newline().unwrap();
        assert_eq!(skipped, 6); // " line\n"

        // Should now be at start of second line
        let mut buf = vec![0u8; 11];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"Second line");
    }

    #[test]
    fn test_skip_to_newline_at_newline() {
        let temp_dir = TempDir::new().unwrap();

        let test_data = b"Line1\nLine2\nLine3\n";
        let file = create_test_file_with_data(temp_dir.path(), "test.dat", test_data).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Seek exactly to newline
        reader.seek(5).unwrap(); // Position at '\n'

        let skipped = reader.skip_to_newline().unwrap();
        assert_eq!(skipped, 1); // Just the newline

        // Should now be at start of Line2
        let mut buf = vec![0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"Line2");
    }

    #[test]
    fn test_skip_to_newline_no_newline() {
        let temp_dir = TempDir::new().unwrap();

        // Create a larger test without newlines to ensure proper EOF handling
        let mut test_data = vec![b'X'; 1024]; // Exactly 2 aligned blocks
        test_data[1023] = b'Y'; // Mark the last byte
        let file = create_test_file_with_data(temp_dir.path(), "test.dat", &test_data).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let skipped = reader.skip_to_newline().unwrap();
        // With Direct I/O, we might read aligned blocks, but position tracking should be accurate
        assert_eq!(reader.position(), 1024); // Should be at end of actual data
        assert_eq!(skipped, 1024); // Should have skipped exactly the data size

        // Next read should return EOF
        let mut buf = vec![0u8; 10];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn test_skip_to_newline_across_buffers() {
        let temp_dir = TempDir::new().unwrap();

        // Create large data that spans multiple buffers
        let mut test_data = vec![b'X'; 100_000];
        test_data[95_000] = b'\n'; // Put newline far into the data
        let file = create_test_file_with_data(temp_dir.path(), "test.dat", &test_data).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let skipped = reader.skip_to_newline().unwrap();
        assert_eq!(skipped, 95_001); // Up to and including the newline

        // Verify position
        assert_eq!(reader.position(), 95_001);
    }

    #[test]
    fn test_seek_trait() {
        let temp_dir = TempDir::new().unwrap();
        let test_size = 8192;
        let (file, test_data) = create_test_file(temp_dir.path(), "test.dat", test_size).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Test SeekFrom::Start - use Seek trait explicitly
        let pos = <AlignedReader as Seek>::seek(&mut reader, SeekFrom::Start(1000)).unwrap();
        assert_eq!(pos, 1000);
        assert_eq!(reader.position(), 1000);

        // Test SeekFrom::Current with positive offset
        let pos = <AlignedReader as Seek>::seek(&mut reader, SeekFrom::Current(500)).unwrap();
        assert_eq!(pos, 1500);
        assert_eq!(reader.position(), 1500);

        // Test SeekFrom::Current with negative offset
        let pos = <AlignedReader as Seek>::seek(&mut reader, SeekFrom::Current(-200)).unwrap();
        assert_eq!(pos, 1300);
        assert_eq!(reader.position(), 1300);

        // Test SeekFrom::End with negative offset
        let pos = <AlignedReader as Seek>::seek(&mut reader, SeekFrom::End(-100)).unwrap();
        assert_eq!(pos, (test_size - 100) as u64);
        assert_eq!(reader.position(), (test_size - 100) as u64);

        // Verify we can read from the seeked position
        let mut buf = vec![0u8; 100];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(buf, &test_data[test_size - 100..]);
    }

    #[test]
    fn test_io_tracker() {
        let temp_dir = TempDir::new().unwrap();
        let test_size = 100 * 1024; // 100KB to ensure multiple reads
        let (file, _) = create_test_file(temp_dir.path(), "test.dat", test_size).unwrap();
        let tracker = IoStatsTracker::new();
        let mut reader = AlignedReader::from_fd_with_tracer(file, Some(tracker.clone())).unwrap();

        // Read some data
        let mut buf = vec![0u8; 1000];
        reader.read_exact(&mut buf).unwrap();

        // Check that I/O was tracked
        let (ops, bytes) = tracker.get_read_stats();
        assert_eq!(ops, 1); // Should be exactly one read operation
        assert!(bytes >= 1000); // Should have read at least the requested bytes

        // Seek far enough to force a new read
        reader.seek(80_000).unwrap();
        reader.read_exact(&mut buf).unwrap();

        // Check updated stats
        let (ops2, bytes2) = tracker.get_read_stats();
        assert_eq!(ops2, 2); // Should now be two read operations
        assert!(bytes2 > bytes);

        // Read a large amount to force multiple buffer refills
        let mut large_buf = vec![0u8; 200_000];
        reader.seek(0).unwrap();
        let n = reader.read(&mut large_buf).unwrap();

        // Check that we had multiple reads
        let (ops3, _) = tracker.get_read_stats();
        assert!(ops3 > 2); // Should have more read operations
        assert_eq!(n, test_size); // Should have read the entire file
    }

    #[test]
    fn test_chunk_reader_interface() {
        let temp_dir = TempDir::new().unwrap();

        // Create test data that's already aligned
        let test_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let file = create_test_file_with_data(temp_dir.path(), "test.dat", &test_data).unwrap();
        let chunk_reader = AlignedChunkReader::new(file).unwrap();

        assert_eq!(chunk_reader.len(), 1024);

        // Test get_bytes
        let bytes = chunk_reader.get_bytes(100, 50).unwrap();
        assert_eq!(&bytes[..], &test_data[100..150]);

        // Test get_read
        let mut reader = chunk_reader.get_read(500).unwrap();
        let mut buf = vec![0u8; 100];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, &test_data[500..600]);
    }

    #[test]
    fn test_chunk_reader_with_tracker() {
        let temp_dir = TempDir::new().unwrap();

        // Create larger test data
        let test_data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let file = create_test_file_with_data(temp_dir.path(), "test.dat", &test_data).unwrap();
        let tracker = IoStatsTracker::new();

        let chunk_reader =
            AlignedChunkReader::new_with_tracker(file, Some(tracker.clone())).unwrap();

        // Read some chunks
        let _ = chunk_reader.get_bytes(0, 1000).unwrap();
        let _ = chunk_reader.get_bytes(50_000, 2000).unwrap();

        // Check that I/O was tracked
        let (ops, bytes) = tracker.get_read_stats();
        assert!(ops > 0);
        assert!(bytes > 0);
    }

    #[test]
    fn test_read_line_basic() {
        let temp_dir = TempDir::new().unwrap();

        let content = "First line\nSecond line\nThird line\n";
        let file =
            create_test_file_with_data(temp_dir.path(), "test_lines.dat", content.as_bytes())
                .unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let mut line = String::new();

        // Read first line
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 11);
        assert_eq!(line, "First line\n");

        // Read second line
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 12);
        assert_eq!(line, "Second line\n");

        // Read third line
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 11);
        assert_eq!(line, "Third line\n");

        // EOF
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 0);
        assert_eq!(line, "");
    }

    #[test]
    fn test_read_line_no_final_newline() {
        let temp_dir = TempDir::new().unwrap();

        let content = "Line without newline";
        let file =
            create_test_file_with_data(temp_dir.path(), "test_no_newline.dat", content.as_bytes())
                .unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let mut line = String::new();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 20);
        assert_eq!(line, "Line without newline");

        // Next read should return EOF
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 0);
    }

    #[test]
    fn test_read_line_long_lines() {
        let temp_dir = TempDir::new().unwrap();

        // Create a line longer than the default buffer size
        let long_line = "x".repeat(100_000);
        let content = format!("{}\nshort\n", long_line);
        let file =
            create_test_file_with_data(temp_dir.path(), "test_long_lines.dat", content.as_bytes())
                .unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let mut line = String::new();

        // Read the long line
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 100_001); // 100_000 x's + newline
        assert_eq!(line.len(), 100_001);
        assert!(line.ends_with('\n'));

        // Read the short line
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 6);
        assert_eq!(line, "short\n");
    }

    #[test]
    fn test_read_line_empty_lines() {
        let temp_dir = TempDir::new().unwrap();

        let content = "\n\nline\n\n";
        let file =
            create_test_file_with_data(temp_dir.path(), "test_empty_lines.dat", content.as_bytes())
                .unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let mut line = String::new();

        // First empty line
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 1);
        assert_eq!(line, "\n");

        // Second empty line
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 1);
        assert_eq!(line, "\n");

        // Line with content
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 5);
        assert_eq!(line, "line\n");

        // Final empty line
        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 1);
        assert_eq!(line, "\n");
    }

    #[test]
    fn test_read_line_crlf() {
        let temp_dir = TempDir::new().unwrap();

        // Windows-style line endings
        let content = "Line 1\r\nLine 2\r\n";
        let file = create_test_file_with_data(temp_dir.path(), "test_crlf.dat", content.as_bytes())
            .unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let mut line = String::new();

        // Should read up to and including \n
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 8);
        assert_eq!(line, "Line 1\r\n");

        line.clear();
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes, 8);
        assert_eq!(line, "Line 2\r\n");
    }

    #[test]
    fn test_direct_io_alignment_edge_cases() {
        let temp_dir = TempDir::new().unwrap();
        // Test various file sizes around alignment boundaries
        let test_cases = [
            (DIRECT_IO_ALIGNMENT - 1, "one byte less than alignment"),
            (DIRECT_IO_ALIGNMENT, "exactly aligned"),
            (DIRECT_IO_ALIGNMENT + 1, "one byte more than alignment"),
            (
                DIRECT_IO_ALIGNMENT * 2 - 1,
                "one byte less than 2x alignment",
            ),
            (DIRECT_IO_ALIGNMENT * 2, "exactly 2x aligned"),
            (1, "single byte"),
            (10, "very small file"),
        ];

        for (idx, (size, description)) in test_cases.iter().enumerate() {
            let data = vec![b'A'; *size];
            let file =
                create_test_file_with_data(temp_dir.path(), &format!("test_{}.dat", idx), &data)
                    .unwrap();

            let mut reader = AlignedReader::from_fd(file).unwrap();
            let mut buf = vec![0u8; size + 100]; // Buffer larger than file

            let bytes_read = reader.read(&mut buf).unwrap();
            assert_eq!(bytes_read, *size, "Failed for {}", description);
            assert_eq!(
                &buf[..*size],
                &data[..],
                "Data mismatch for {}",
                description
            );

            // Verify EOF behavior
            let bytes_read = reader.read(&mut buf).unwrap();
            assert_eq!(bytes_read, 0, "EOF failed for {}", description);
        }
    }

    #[test]
    fn test_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let file = create_test_file_with_data(temp_dir.path(), "empty.dat", &[]).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Test read
        let mut buf = vec![0u8; 100];
        let bytes_read = reader.read(&mut buf).unwrap();
        assert_eq!(bytes_read, 0);

        // Test read_line
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line).unwrap();
        assert_eq!(bytes_read, 0);
        assert_eq!(line, "");

        // Test position
        assert_eq!(reader.position(), 0);
    }

    #[test]
    fn test_refill_buffer_at_eof() {
        let temp_dir = TempDir::new().unwrap();

        // Create a file smaller than buffer size
        let data = vec![b'X'; 100];
        let file =
            create_test_file_with_data(temp_dir.path(), "test_refill_eof.dat", &data).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Read all data
        let mut buf = vec![0u8; 200];
        let bytes_read = reader.read(&mut buf).unwrap();
        assert_eq!(bytes_read, 100);

        // Multiple reads at EOF should consistently return 0
        for _ in 0..5 {
            let bytes_read = reader.read(&mut buf).unwrap();
            assert_eq!(bytes_read, 0);
        }

        // Position should remain stable
        assert_eq!(reader.position(), 100);
    }

    #[test]
    fn test_seek_beyond_eof_then_read() {
        let temp_dir = TempDir::new().unwrap();

        let data = vec![b'A'; 1000];
        let file =
            create_test_file_with_data(temp_dir.path(), "test_seek_beyond.dat", &data).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Seek beyond EOF
        reader.seek(2000).unwrap();

        // Read should return 0
        let mut buf = vec![0u8; 100];
        let bytes_read = reader.read(&mut buf).unwrap();
        assert_eq!(bytes_read, 0);

        // After reading at EOF, position should stay at EOF (1000)
        // The position is corrected when we detect EOF during refill_buffer
        assert_eq!(reader.position(), 1000);

        // Seek back and read should work
        reader.seek(500).unwrap();
        let bytes_read = reader.read(&mut buf).unwrap();
        assert_eq!(bytes_read, 100);
        assert_eq!(&buf[..100], &data[500..600]);
    }

    #[test]
    fn test_position_after_partial_buffer_read() {
        let temp_dir = TempDir::new().unwrap();

        // Create file larger than buffer
        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let file = create_test_file_with_data(temp_dir.path(), "test_position.dat", &data).unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        // Read small amount
        let mut buf = vec![0u8; 100];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(reader.position(), 100);

        // Read more within same buffer
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(reader.position(), 200);

        // Seek and verify position
        reader.seek(70_000).unwrap();
        assert_eq!(reader.position(), 70_000);

        // Read and verify position updates correctly
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(reader.position(), 70_100);
    }

    #[test]
    fn test_utf8_in_read_line() {
        let temp_dir = TempDir::new().unwrap();

        // Create file with UTF-8 content including multi-byte characters
        let content = "Hello, 世界!\nЗдравствуй, мир!\n🦀 Rust 🦀\n";
        let file = create_test_file_with_data(temp_dir.path(), "test_utf8.dat", content.as_bytes())
            .unwrap();
        let mut reader = AlignedReader::from_fd(file).unwrap();

        let mut line = String::new();

        // Read first line with Chinese characters
        let bytes = reader.read_line(&mut line).unwrap();
        assert_eq!(line, "Hello, 世界!\n");
        assert_eq!(bytes, "Hello, 世界!\n".len());

        // Read second line with Cyrillic
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert_eq!(line, "Здравствуй, мир!\n");

        // Read third line with emojis
        line.clear();
        reader.read_line(&mut line).unwrap();
        assert_eq!(line, "🦀 Rust 🦀\n");
    }

    #[test]
    fn test_with_start_position_edge_cases() {
        let temp_dir = TempDir::new().unwrap();

        let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        let file =
            create_test_file_with_data(temp_dir.path(), "test_start_pos.dat", &data).unwrap();

        // Test starting at various positions
        let test_positions = vec![0, 1, 511, 512, 512, 63999, 64000, 64001, 99999, 100000];

        for start_pos in test_positions {
            let mut reader = AlignedReader::from_fd_with_start_position(
                file.clone(),
                start_pos,
                None, // No IoStatsTracker for this test
            )
            .unwrap();

            // Read and verify data
            let mut buf = vec![0u8; 100];
            let bytes_to_read = std::cmp::min(100, data.len() - start_pos as usize);
            let bytes_read = reader.read(&mut buf).unwrap();
            assert_eq!(bytes_read, bytes_to_read);
            assert_eq!(
                &buf[..bytes_read],
                &data[start_pos as usize..start_pos as usize + bytes_read]
            );

            // Now verify position after read
            assert_eq!(reader.position(), start_pos + bytes_read as u64);
        }
    }
}
