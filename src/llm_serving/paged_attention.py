"""
Educational Simulation of PagedAttention and Block-Based Memory Management.

This module simulates the core concepts of vLLM's PagedAttention:
1. A Physical Cache Pool (non-contiguous memory blocks).
2. A Block Table (mapping logical sequence positions to physical blocks).
3. On-Demand Allocation (allocating one block at a time).
4. Paged Attention Math (computing attention across scattered blocks).
"""

import torch

class BlockAllocator:
    """
    Simulates the Memory Manager (like an OS Virtual Memory Manager).
    Includes Block-Level Sharing and Copy-on-Write (CoW) for Prefix Caching.
    """
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        self.physical_k_cache = torch.zeros((num_blocks, block_size, head_dim))
        self.physical_v_cache = torch.zeros((num_blocks, block_size, head_dim))
        
        self.free_blocks = list(range(num_blocks))
        
        # Block Table: maps seq_id -> list of physical block indices
        self.block_tables: dict[str, list[int]] = {}
        self.seq_lengths: dict[str, int] = {}
        
        # Reference Counter for Block-Level Sharing
        # Maps physical_block_id -> number of sequences pointing to it
        self.ref_counts = {i: 0 for i in range(num_blocks)}

    def allocate_sequence(self, seq_id: str, share_from_seq: str = None):
        """
        Initialize a new sequence. 
        If share_from_seq is provided, it performs Block-Level Sharing by 
        pointing to the exact same physical blocks and incrementing ref counts.
        """
        if share_from_seq and share_from_seq in self.block_tables:
            # BLOCK SHARING: Point to the exact same physical blocks
            shared_blocks = list(self.block_tables[share_from_seq])
            self.block_tables[seq_id] = shared_blocks
            self.seq_lengths[seq_id] = self.seq_lengths[share_from_seq]
            
            # Increment reference counts for shared blocks
            for block_id in shared_blocks:
                self.ref_counts[block_id] += 1
            print(f"[Prefix Cache] {seq_id} is sharing {len(shared_blocks)} blocks from {share_from_seq}.")
        else:
            if len(self.free_blocks) == 0:
                raise RuntimeError("Out of Memory! No free blocks available.")
            
            physical_block_id = self.free_blocks.pop(0)
            self.block_tables[seq_id] = [physical_block_id]
            self.seq_lengths[seq_id] = 0
            self.ref_counts[physical_block_id] = 1

    def append_token(self, seq_id: str, k_vector: torch.Tensor, v_vector: torch.Tensor):
        """Simulate appending a newly generated token with Copy-on-Write (CoW)."""
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)
            
        current_length = self.seq_lengths[seq_id]
        logical_block_idx = current_length // self.block_size
        offset_within_block = current_length % self.block_size
        
        # Check if we need to allocate a brand new block
        if current_length > 0 and offset_within_block == 0:
            if len(self.free_blocks) == 0:
                raise RuntimeError("Out of Memory! Cannot allocate new block.")
            new_block_id = self.free_blocks.pop(0)
            self.block_tables[seq_id].append(new_block_id)
            self.ref_counts[new_block_id] = 1
            physical_block_id = new_block_id
        else:
            physical_block_id = self.block_tables[seq_id][logical_block_idx]
            
            # COPY-ON-WRITE (CoW) Mechanism
            # If we are trying to write to a block that is shared by someone else...
            if self.ref_counts[physical_block_id] > 1:
                if len(self.free_blocks) == 0:
                    raise RuntimeError("OOM during Copy-on-Write!")
                
                # 1. Grab a new empty block
                new_cow_block_id = self.free_blocks.pop(0)
                
                # 2. Copy the existing data over
                self.physical_k_cache[new_cow_block_id] = self.physical_k_cache[physical_block_id].clone()
                self.physical_v_cache[new_cow_block_id] = self.physical_v_cache[physical_block_id].clone()
                
                # 3. Update block tables and reference counts
                self.block_tables[seq_id][logical_block_idx] = new_cow_block_id
                self.ref_counts[physical_block_id] -= 1
                self.ref_counts[new_cow_block_id] = 1
                
                physical_block_id = new_cow_block_id
                print(f"[CoW Triggered] {seq_id} modified a shared block. Cloned to Block {new_cow_block_id}.")
        
        # Finally, write the new token
        self.physical_k_cache[physical_block_id, offset_within_block] = k_vector
        self.physical_v_cache[physical_block_id, offset_within_block] = v_vector
        self.seq_lengths[seq_id] += 1

    def free_sequence(self, seq_id: str):
        """Free a sequence by decrementing ref counts. Only return to free pool if ref_count == 0."""
        if seq_id in self.block_tables:
            for block_id in self.block_tables[seq_id]:
                self.ref_counts[block_id] -= 1
                if self.ref_counts[block_id] == 0:
                    self.free_blocks.append(block_id)
            del self.block_tables[seq_id]
            del self.seq_lengths[seq_id]

    def get_fragmentation_stats(self):
        """Calculate how much memory is wasted."""
        total_blocks_used = self.num_blocks - len(self.free_blocks)
        if total_blocks_used == 0:
            return 0.0
            
        total_capacity_allocated = total_blocks_used * self.block_size
        actual_tokens_stored = sum(self.seq_lengths.values())
        
        # The only waste in PagedAttention is the empty slots at the end of the very last block (Internal Fragmentation)
        wasted_slots = total_capacity_allocated - actual_tokens_stored
        wasted_pct = (wasted_slots / total_capacity_allocated) * 100
        
        return {
            "total_capacity_allocated": total_capacity_allocated,
            "actual_tokens_stored": actual_tokens_stored,
            "wasted_slots": wasted_slots,
            "internal_fragmentation_pct": wasted_pct
        }


def simulated_paged_attention(
    query: torch.Tensor, 
    seq_id: str, 
    allocator: BlockAllocator
):
    """
    A pure-Python simulation of how PagedAttention calculates scores.
    Instead of Q * K^T on one giant contiguous tensor, it iterates through
    the scattered physical blocks mapped in the block table.
    
    query shape: [1, head_dim]
    """
    block_table = allocator.block_tables[seq_id]
    seq_len = allocator.seq_lengths[seq_id]
    
    all_scores = []
    
    # 1. Compute Attention Scores Block-by-Block
    for logical_idx, physical_block_id in enumerate(block_table):
        k_block = allocator.physical_k_cache[physical_block_id] # [block_size, head_dim]
        
        # If it's the last block, it might not be completely full
        if logical_idx == len(block_table) - 1:
            tokens_in_block = seq_len % allocator.block_size
            if tokens_in_block == 0 and seq_len > 0:
                tokens_in_block = allocator.block_size
            k_block = k_block[:tokens_in_block]
            
        # Standard attention math: Q * K^T
        # query: [1, head_dim], k_block.T: [head_dim, num_tokens] -> score: [1, num_tokens]
        score = torch.matmul(query, k_block.T) 
        all_scores.append(score)
        
    # 2. Concatenate scattered scores and apply Softmax
    full_scores = torch.cat(all_scores, dim=-1) / (allocator.head_dim ** 0.5)
    attention_weights = torch.softmax(full_scores, dim=-1)
    
    # 3. Multiply with Values Block-by-Block (simplified for simulation)
    # In a real CUDA kernel, step 1, 2, and 3 are fused to avoid materializing full_scores
    output = torch.zeros_like(query)
    current_token_idx = 0
    
    for logical_idx, physical_block_id in enumerate(block_table):
        v_block = allocator.physical_v_cache[physical_block_id]
        
        tokens_in_block = allocator.block_size
        if logical_idx == len(block_table) - 1:
            tokens_in_block = seq_len % allocator.block_size
            if tokens_in_block == 0 and seq_len > 0:
                tokens_in_block = allocator.block_size
                
        v_block = v_block[:tokens_in_block]
        
        # Extract the weights for just this block
        block_weights = attention_weights[:, current_token_idx : current_token_idx + tokens_in_block]
        
        # out = weights * V
        output += torch.matmul(block_weights, v_block)
        current_token_idx += tokens_in_block
        
    return output
