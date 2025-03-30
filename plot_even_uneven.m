close all
clear all
clc

n_even = linspace(2,10,5);
n_uneven = linspace(1,9,5);
un_even = [-exp(1) 4*exp(1) -31*exp(1) 379*exp(1) -6556*exp(1)];
un_uneven = [0 0 0 0 0];
%%
figure; 

plot(n_even ,abs(un_even),'s-','LineWidth',2,'Color',[1.0000, 0.4980, 0.0549])
hold on
plot(n_uneven, un_uneven, 's-','LineWidth',2,'Color',[0.1216, 0.4667, 0.7059])
grid on
title("Plot of (n'th) derivative of e^{cos(x)} for x = 0")
xlabel("n")
ylabel("abs(u^{(n)}(0))")
H = gca; 
H.LineWidth = 1;
H.FontSize = 12;
xlim([-0.5 10.5])
ylim([-500*exp(1) 6900*exp(1)])
legend(["Even" "Uneven"])

