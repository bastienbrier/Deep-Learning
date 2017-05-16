function W_i2i = make_proper(W_i2i);
W_i2i    = W_i2i - diag(diag(W_i2i));
W_i2i    = (W_i2i + W_i2i')/2;