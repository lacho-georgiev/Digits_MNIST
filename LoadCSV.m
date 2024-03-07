% data = load('MNIST/mnist_test.csv');
data = load('MNIST/mnist_train.csv');

label = data(:, 1)';
images = data(:, 2:785);
labels = zeros(10, length(label));

for i = 1: length(labels)
    labels(label(1, i) + 1, i) = 1;
    images(i, :) = images(i, :)./255;
end

labels = labels';
