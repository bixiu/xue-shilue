function result = cond_ent(Y, X)
% Calculates the conditional entropy of y given x
result = 0;

tab = tabulate(X);   %统计变量出现次数

% Remove zero-entries
tab = tab(tab(:,3)~=0,:);
%disp(tab);
for i = 1:size(tab,1)
    % Get entropy for y values where x is the current value
    H = ent(Y(X == tab(i,1)));
    % Get probability
    prob = tab(i, 3) / 100;
    % Combine
    result = result + prob * H;
end
