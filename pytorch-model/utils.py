import torch


def create_square_causality_mask(sequence_length, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    mask = (torch.triu(torch.ones((sequence_length, sequence_length),
            device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, PAD_IDX, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = create_square_causality_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
