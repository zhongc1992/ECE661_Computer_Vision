function K = IntrinsicSolve(w_mat)
%Calculate Intrinsic matrx K
y0 = (w_mat(2)*w_mat(4)-w_mat(1)*w_mat(5))/(w_mat(1)*w_mat(3)-w_mat(2)^2); % Principle point y
Lambda = w_mat(6)-(w_mat(4)^2+y0*(w_mat(2)*w_mat(4)-w_mat(1)*w_mat(5)))/w_mat(1);
alpha_x = sqrt(Lambda/w_mat(1)); % Scale factor in x
alpha_y = sqrt(Lambda*w_mat(1)/(w_mat(1)*w_mat(3)-w_mat(2)^2)); % Scale factor in y
s = -w_mat(2)*alpha_x^2*alpha_y/Lambda; % Skewness factor
x0 = s*y0/alpha_y-w_mat(4)*alpha_x^2/Lambda; % Principle point x
K = [alpha_x s x0; 0 alpha_y y0; 0 0 1]; % Intrinsic parameter matrix K

end