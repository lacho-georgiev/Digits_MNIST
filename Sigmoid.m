% function sig = Sigmoid(z)
% 
%     for i = 1: length(z)
%         sig(i) = 1 / (1 + exp(-z(1, i)));
%     end
% 
% end

function s = Sigmoid(z)
    s = 1./(1+exp(-z));
end
