import torch
from gpu_memory import *

def rect_naive_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in)//lsb)*lsb - lsb
    v_rp = (torch.max(array_in)//lsb)*lsb + lsb
    #print(f"v_rn: {v_rn}")
    #print(f"v_rp: {v_rp}")
    ramp = torch.linspace(v_rn, v_rp, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp = ramp.view(ramp.shape[0],1,1,1)
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, int((v_rp-v_rn)/lsb + 1), device = cuda).type(torch.int16)
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1)
    # create array of comparator outputs
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next)
    # extract row and column tensors for all samples and ramp values
    row = comp.any(dim=2)
    col = comp.any(dim=1)
    # free some space
    del comp, array_in
    torch.cuda.empty_cache()
    # naive valid matrix for no conflicts
    valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1))
    valid = torch.einsum("ij->ji", (valid,))
    # you have to use torch.matmul to reconstruct array
    # rearrange row and column in correct order for matmul
    row = torch.einsum("ijk->kij", (row,))
    col = torch.einsum("ijk->kij", (col,))
    row = row.view(row.shape[0],row.shape[1],row.shape[2],1)
    col = col.view(col.shape[0], col.shape[1], 1, col.shape[2])
    # array reconstructed from row and column
    array_out = torch.matmul(row.float(), col.float())
    # free some space
    del row, col
    torch.cuda.empty_cache()
    # go back to all channels in one vector
    naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch,ramp.shape[0],array[0]*array[1])
    # reshape valid for torch.mul
    valid = valid.view(valid.shape[0], valid.shape[1], 1)
    # use torch.mul to get rid of invalid arguments
    naive_valid = torch.mul(naive_out,valid.float())
    # free some space
    del naive_out
    torch.cuda.empty_cache()
    # reshape ramp for torch.einsum
    ramp = ramp.view(ramp.shape[0])
    # use torch.einsum to sum across all ramp values and get decoded output
    naive_decoded = torch.einsum("ijk,j->ik", (naive_valid,ramp.float()))
    return naive_decoded

def rect_naive_interleaved_tensor(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    # generate rampup and rampdown for comparators and add dimentions for comparison
    n_channels = array[0]*array[1]
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in)//lsb)*lsb - lsb
    v_rp = (torch.max(array_in)//lsb)*lsb + lsb
    v_r = max(abs(v_rn),abs(v_rp))
    rampup = torch.linspace(-v_r, v_r, int(2*v_r/lsb + 1), device = cuda).type(torch.int16)
    rampup = rampup.view(rampup.shape[0],1,1,1)
    rampup_next = torch.linspace(-v_r+lsb, v_r+lsb, int(2*v_r/lsb) + 1, device = cuda).type(torch.int16)
    rampup_next = rampup_next.view(rampup_next.shape[0],1,1,1)
    rampdown = torch.linspace(v_r, -v_r, int(2*v_r/lsb + 1), device = cuda).type(torch.int16)
    rampdown = rampdown.view(rampdown.shape[0],1,1,1)
    rampdown_next = torch.linspace(v_r+lsb, -v_r+lsb, int(2*v_r/lsb + 1), device = cuda).type(torch.int16)
    rampdown_next = rampdown_next.view(rampdown_next.shape[0],1,1,1)
    # create mask for up/down, create two comp tensors, combine them for validation but decode them separately
    maskup = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
    lin_maskup = torch.arange(0, n_channels, 2)
    x_maskup = lin_maskup % array[0]
    y_maskup = lin_maskup // array[0]
    maskup[x_maskup, y_maskup] = 1
    maskdown = torch.zeros(array[0], array[1], 1, device=cuda, dtype=torch.int16)
    lin_maskdown = torch.arange(1, n_channels, 2)
    x_maskdown = lin_maskdown % array[0]
    y_maskdown = lin_maskdown // array[0]
    maskdown[x_maskdown, y_maskdown] = 1
    # create tensor of comparator outputs
    compup = torch.ge(array_in*maskup,rampup) & torch.lt(array_in*maskup,rampup_next)
    compdown = torch.ge(array_in * maskdown, rampdown) & torch.lt(array_in * maskdown, rampdown_next)
    # extract row and column tensors for all samples and ramp values, both for up and down
    rowup = compup.any(dim=2)
    colup = compup.any(dim=1)
    rowdown = compdown.any(dim=2)
    coldown = compdown.any(dim=1)
    # combine row and column to get valid tensor
    row = rowup|rowdown
    col = colup|coldown
    # free some space
    del compup, compdown, array_in, rowup, rowdown,
    torch.cuda.empty_cache()
    # naive valid matrix for no conflicts
    valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1))
    valid = torch.einsum("ij->ji", (valid,))
    # you have to use torch.matmul to reconstruct array
    # rearrange row and column in correct order for matmul
    row = torch.einsum("ijk->kij", (row,))
    col = torch.einsum("ijk->kij", (col,))
    row = row.view(row.shape[0],row.shape[1],row.shape[2],1)
    col = col.view(col.shape[0], col.shape[1], 1, col.shape[2])
    # array reconstructed from row and column
    array_out = torch.matmul(row.float(), col.float())
    # free some space
    del row, col
    torch.cuda.empty_cache()
    # go back to all channels in one vector
    naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch,rampup.shape[0],array[0]*array[1])
    maskup_out = maskup[idx_dim0, idx_dim1,0].view(1,1,n_channels)
    maskdown_out = maskdown[idx_dim0, idx_dim1,0].view(1,1,n_channels)
    # reshape valid for torch.mul
    valid = valid.view(valid.shape[0], valid.shape[1], 1)
    # use torch.mul to get rid of invalid arguments
    naive_valid = torch.mul(naive_out,valid.float())
    # free some space
    del naive_out
    torch.cuda.empty_cache()
    # reshape ramp for torch.einsum
    rampup = rampup.view(rampup.shape[0])
    rampdown = rampdown.view(rampup.shape[0])
    #maskup = torch.zeros(1,1,array[0]*array[1], device=cuda, dtype=torch.int16)
    #maskup[0,0,lin_maskup] = 1
    #maskdown = torch.zeros(1, 1, array[0] * array[1], device=cuda, dtype=torch.int16)
    #maskdown[0,0,lin_maskdown] = 1
    # use torch.einsum to sum across all ramp values and get decoded output
    naive_decoded = torch.einsum("ijk,j->ik", (naive_valid*maskup_out.float(),rampup.float())) + \
                    torch.einsum("ijk,j->ik", (naive_valid*maskdown_out.float(),rampdown.float()))
    return naive_decoded

def rect_naive_tensor_profile(array, array_in, num_bits, v_ref, samples_batch, idx_dim0, idx_dim1, cuda):
    # generate ramp for comparators and add dimentions for comparison
    lsb = 2*v_ref/2**num_bits
    v_rn = (torch.min(array_in)//lsb)*lsb - lsb
    v_rp = (torch.max(array_in)//lsb)*lsb + lsb
    ramp = torch.linspace(v_rn, v_rp, (v_rp-v_rn)/lsb + 1, device = cuda).type(torch.int16)
    ramp = ramp.view(ramp.shape[0],1,1,1)
    ramp_next = torch.linspace(v_rn+lsb, v_rp+lsb, (v_rp-v_rn)/lsb + 1, device = cuda).type(torch.int16)
    ramp_next = ramp_next.view(ramp_next.shape[0],1,1,1)
    # create array of comparator outputs
    comp = torch.ge(array_in,ramp) & torch.lt(array_in,ramp_next)
    # extract row and column tensors for all samples and ramp values
    row = comp.any(dim=2)
    col = comp.any(dim=1)
    # free some space
    del array_in
    torch.cuda.empty_cache()
    # naive valid matrix for no conflicts
    valid = (torch.le(torch.sum(row,dim=1),1) | torch.le(torch.sum(col,dim=1),1))
    valid = torch.einsum("ij->ji", (valid,))
    # you have to use torch.matmul to reconstruct array
    # rearrange row and column in correct order for matmul
    row = torch.einsum("ijk->kij", (row,))
    col = torch.einsum("ijk->kij", (col,))
    row = row.view(row.shape[0], row.shape[1], row.shape[2], 1)
    col = col.view(col.shape[0], col.shape[1], 1, col.shape[2])
    # array reconstructed from row and column
    array_out = torch.matmul(row.float(), col.float())
    # free some space
    del row, col
    torch.cuda.empty_cache()
    # go back to all channels in one vector
    naive_out = array_out[:, :, idx_dim0, idx_dim1].view(samples_batch, ramp.shape[0], array[0] * array[1])
    # reshape valid for torch.mul
    valid = valid.view(valid.shape[0], valid.shape[1], 1)
    # use torch.mul to get rid of invalid arguments
    naive_valid = torch.mul(naive_out, valid.float())
    # get data activity
    bit_tx_batch = torch.sum(naive_valid)/samples_batch
    return bit_tx_batch