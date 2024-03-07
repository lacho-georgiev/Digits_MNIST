function [dn_w, dn_b] = BackPropagate(x, y, weights, biases, L)
    a = cell(1, L);
    a{1} = x;
    for l = 1: L - 1
        z{l + 1} = a{l} * weights{l} +  biases{l};
        a{l + 1} = Sigmoid(z{l + 1});
    end

    d{L} = (a{L} - y) .* SigmoidPrime(z{L});
    
    for l = L - 1: -1: 2
        d{l} = (d{l + 1} * weights{l}') .* SigmoidPrime(z{l});
    end
    
    dn_b = cell(1, L);
    dn_w = cell(1, L);
    
    for l = L : -1: 2
        dn_b{l - 1} = d{l};
        dn_w{l - 1} = a{l - 1}' * d{l};
    end
end