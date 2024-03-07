% function sigp = SigmoidPrime(z)
% 
%     for i = 1: length(z)
%         sigp(i) = Sigmoid(z(1, i)) * (1 - Sigmoid(z(1, i)));
%     end
% 
% end

function s = SigmoidPrime(z)
    s = Sigmoid(z).*(1-Sigmoid(z));
end
