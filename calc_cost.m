function cost = calc_cost(images, labels, w, b, L)
    cost = 0;
    for i = 1 : length(images)
        out = FeedForward(images(i, :), w, b, L);
        cost = cost + sum((labels(i, :) - out).^2);
    end
    cost = cost / length(images);
end

