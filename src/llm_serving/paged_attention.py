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
    Instead of reserving max_seq_len for every request, it holds a pool of
    fixed-size blocks and doles them out on demand.
    """
    def __init__(self, num_blocks: int, block_size: int, head_dim: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.head_dim = head_dim
        
        # The actual physical GPU memory. 
        # Shape: [num_blocks, block_size, head_dim]
        # In a real system, there would be separate pools for Keys and Values.
        self.physical_k_cache = torch.zeros((num_blocks, block_size, head_dim))
        self.physical_v_cache = torch.zeros((num_blocks, block_size, head_dim))
        
        # Keep track of which physical blocks are free
        self.free_blocks = list(range(num_blocks))
        
        # Block Table: maps seq_id -> list of physical block indices
        self.block_tables: dict[str, list[int]] = {}
        
        # Track how many tokens are currently in the last block of a sequence
        self.seq_lengths: dict[str, int] = {}

    def allocate_sequence(self, seq_id: str):
        """Initialize a new sequence with 1 physical block."""
        if len(self.free_blocks) == 0:
            raise RuntimeError("Out of Memory! No free blocks available.")
        
        physical_block_id = self.free_blocks.pop(0)
        self.block_tables[seq_id] = [physical_block_id]
        self.seq_lengths[seq_id] = 0
        return physical_block_id

    def append_token(self, seq_id: str, k_vector: torch.Tensor, v_vector: torch.Tensor):
        """
        Simulate appending a newly generated token to the cache.
        Allocates a new block dynamically ONLY if the current block is full.
        """
        if seq_id not in self.block_tables:
            self.allocate_sequence(seq_id)
            
        current_length = self.seq_lengths[seq_id]
        
        # Check if the last block is full
        if current_length > 0 and current_length % self.block_size == 0:
            if len(self.free_blocks) == 0:
                raise RuntimeError("Out of Memory! Cannot allocate new block.")
            new_block_id = self.free_blocks.pop(0)
            self.block_tables[seq_id].append(new_block_id)
            
        # Figure out where to write in physical memory
        logical_block_idx = current_length // self.block_size
        offset_within_block = current_length % self.block_size
        
        physical_block_id = self.block_tables[seq_id][logical_block_idx]
        
        # Write the new KV vectors into the non-contiguous physical cache
        self.physical_k_cache[physical_block_id, offset_within_block] = k_vector
        self.physical_v_cache[physical_block_id, offset_within_block] = v_vector
        
        self.seq_lengths[seq_id] += 1

    def free_sequence(self, seq_id: str):
        """When a request finishes, instantly return its blocks to the free pool."""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
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
