%% Load all relevant parameters

clear
clc
close all

% Problem parameters
epsilon = 0.1;           % Diffusion coefficient
T_final = 100;           % Final simulation time
alpha = [1, 4, 16];      % Given alpha values
a = [1, 1, 1];           % Given a_i values
b = [0, 0, 0];           % Given b_i values

%% Spatial convergence

% grid sizes for study
N_values = [20, 40, 80, 160];

% empty error vector
errors_space = zeros(length(N_values), 1);

% for each number of grid sizes
for idx = 1:length(N_values)
    N = N_values(idx);          % choose N from list
    h = 2 / N;                  % compute h for given grid size
    x = linspace(-1, 1, N+1);   % create spatial grid  
    
    k = h^2 / (2*epsilon);      % stability constraint 
    M = ceil(T_final / k);      % time grid size from k    
    t = linspace(0, T_final, M+1); % create temporal grid
    
    u = exact_solution(x, 0, alpha, a, b, epsilon);
    
    % Do the FCTS scheme in spatial domain
    for n = 1:M
        u(1) = exact_solution(x(1), t(n), alpha, a, b, epsilon); % fix BCs
        u(end) = exact_solution(x(end), t(n), alpha, a, b, epsilon); 
        
        u_new = u;
        for i = 2:N
            u_new(i) = u(i) + epsilon * k / h^2 * (u(i+1) - 2*u(i) + u(i-1));
        end
        u = u_new;
    end
    
    u_exact = exact_solution(x, T_final, alpha, a, b, epsilon); % exact solution
    errors_space(idx) = max(abs(u - u_exact)); % LTE
end

% h reference (order h^2)
hs = 2./N_values; 
href = hs.^2;

% plotting
figure;
loglog(hs, errors_space, 'o-b', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('Spatial step size h');
ylabel('Max Error');
title('Convergence test of diffusion-heat system');
hold on;
h = gca; 
h.LineWidth = 1.5; 
h.FontSize = 16; 
%loglog(N_values, errors_space(1)*(N_values(1)./N_values).^2, '--');
loglog(hs,1/(0.5e5)*href,'--k')
legend('Error','O(h^2) reference','Location','southeast');

%% Time convergence
N = 160;  % fixed spatial grid fineness (need to "fine enough")
h = 2/N;  % compute step size h
x = linspace(-1, 1, N+1); % contruct spatial grid

% we use a list of sigmas which will be used to scale the system down 
% such that the stability constraint is kept
sigma_values = [1.0 0.8, 0.4, 0.2 0.1 0.05]; % these must be <= 1 > 0 
errors_time = zeros(length(sigma_values), 1);% empty error vector
k_values = zeros(length(sigma_values), 1);   % empty k-vector

% perform the FTCS scheme in time domain
for idx = 1:length(sigma_values)
    sigma = sigma_values(idx); % choose the iterations sigma 
    k = sigma * h^2 / (2*epsilon); % stability constraint
    k_values(idx) = k;  % used in the plot
    M = ceil(T_final / k);  
    %k = T_final / M;  % Adjust k to fit exactly into T_final
    t = linspace(0, T_final, M+1);
    
    u = exact_solution(x, 0, alpha, a, b, epsilon);
    
    for n = 1:M
        u(1) = exact_solution(x(1), t(n), alpha, a, b, epsilon); % fix BCs
        u(end) = exact_solution(x(end), t(n), alpha, a, b, epsilon);
        
        u_new = u;
        for i = 2:N
            u_new(i) = u(i) + epsilon * k / h^2 * (u(i+1) - 2*u(i) + u(i-1));
        end
        u = u_new;
    end
    
    u_exact = exact_solution(x, T_final, alpha, a, b, epsilon);
    errors_time(idx) = max(abs(u - u_exact)); % max error
end

% Plot corrected time convergence
figure;
loglog(k_values, errors_time, 'o-r', 'LineWidth', 2, 'MarkerSize', 8);
grid on;
xlabel('Time step size k');
ylabel('Max Error');
title('Time convergence of heat-diffusion system');
hold on;
loglog(k_values, (4e-6)*k_values, '--k'); % needed to scale down the ref
legend('Error','downscaled O(k) reference','Location','southwest');
h = gca; 
h.LineWidth = 1.5; 
h.FontSize = 16; 

%% exact solution function
function u = exact_solution(x, t, alpha, a, b, epsilon)
%parameter list; 
%x is the spatial grid 
%t is the temporal grid
%alpha is an array of alphas
%a and b are the coefficients in the oscillating terms
%epsilon is the diffusion constant

%program loads the exact solution for the system and is used for BC's, etc.

    u = zeros(size(x));
    for i = 1:length(alpha)
        u = u + exp(-epsilon*alpha(i)^2*t) .* (a(i)*cos(alpha(i)*x) + b(i)*sin(alpha(i)*x));
    end
end
