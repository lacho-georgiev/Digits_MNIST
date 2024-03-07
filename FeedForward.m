function aL = FeedForward(x, weights, biases, L)
    a{1} = x;
    for l = 1: L - 1
        z{l + 1} = a{l} * weights{l} +  biases{l};
        a{l + 1} = Sigmoid(z{l + 1});
    end
    aL = a{L};
end