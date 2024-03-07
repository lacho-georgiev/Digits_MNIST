layers = [784, 28, 10];
L = length(layers);
epochs = 20;
m = 10;
n = 3;
LoadCSV;
weights = cell(1, L - 1);
biases = cell(1, L - 1);
z = cell(1, L);
a = cell(1, L);
d = cell(1, L);
n_w = cell(1, L - 1);
n_b = cell(1, L - 1);
for l = 1: L-1
    n_w{l} = zeros(layers(l), layers(l + 1));
    n_b{l} = zeros(1, layers(l + 1));
end

for l = 1: L - 1
    weights{l} = randn(layers(l), layers(l + 1));
    biases{l} = randn(1, layers(l + 1));
end

for epoch = 1 : epochs
    perm_indx = randperm(length(images));
    images = images(perm_indx, :);
    labels = labels(perm_indx, :);
    
    for batch = 1: length(images)/m
        for l = 1: L-1
            n_w{l} = zeros(layers(l), layers(l + 1));
            n_b{l} = zeros(1, layers(l + 1));
        end
        for mini_batch = (batch - 1) * m + 1 : batch * m
            [dn_w, dn_b] = BackPropagate(images(mini_batch, :), labels(mini_batch, :), weights, biases, L);
            for l = 1: L-1
                n_w{l} = n_w{l} + dn_w{l};
                n_b{l} = n_b{l} + dn_b{l};
            end
        end

        for l = 1: L-1
            biases{l} = biases{l} - (n/m) * n_b{l};
            weights{l} = weights{l} - (n/m) * n_w{l};            
        end
    end
    cost = calc_cost(images, labels, weights, biases, L);
    p = epoch + ": " + cost;
    disp(p);
end

