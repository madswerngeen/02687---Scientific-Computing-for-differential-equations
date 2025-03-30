% Define x range
x = linspace(0,1,10000);

% Compute p(x)
p = 79.1015625 * abs( ...
    (x.*cosh(3.75*x - 0.9375) - 0.25*cosh(3.75*x - 0.9375) + 0.75*sinh(3.75*x - 0.9375)) ...
    .* (2.*cosh(3.75*x - 0.9375).^2 - 3) ...
) ./ abs(cosh(3.75*x - 0.9375)).^5;

% Compute q(x)
q = 59.3261719 * abs( ...
    tanh(3.75*x - 0.9375).*cosh(3.75*x - 0.9375).^2 - 3.*tanh(3.75*x - 0.9375) ...
) ./ abs(cosh(3.75*x - 0.9375)).^4;

% Plot both functions
figure;
plot(x, p, 'b', 'LineWidth', 1.5); hold on;
plot(x, q, 'r', 'LineWidth', 1.5);
xlabel('x');
ylabel('Function values');
title('Plots of p(x) and q(x)');
legend('p(x)', 'q(x)');
h = gca; 
h.LineWidth = 2; 
h.FontSize = 16; 
grid on;
