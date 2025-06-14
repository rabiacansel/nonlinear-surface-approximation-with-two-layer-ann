%31/05/2025

clear all; close all; clc;
logsig = @(x) 1 ./ (1 + exp(-x));
tic
[X, Y] = meshgrid(-5:1:5, -5:1:5);
X_base = [X(:), Y(:)];
targets = [
    -4 4;
    -5 4;
    -3 0;
    -2 0;
    -4 1.5;
    -3.5 4;
    -2.5 4;
    -2.5 3;
    -3 3;
    -4.5 4.5;
    -1.5 2.5;
    -3.5 4;
    -3.5 3.5;
    -2 2.5;
    -2.5 3;
    -1.5 2.5;
    -2 3;
    -2 3.3;
    -2.5 3.5;
    -2 -0.5;
    -2 -1;
    -2 -1.5;
    -2 2;
    3 2.5;
    3 2;
    -5 0
];
n_extra = 50;
noise_std = 0.1;
extra_points = [];
for i = 1:size(targets,1)
    center = targets(i,:);
    jittered = center + randn(n_extra, 2) * noise_std;
    extra_points = [extra_points; jittered];
end
X_base = [X_base; extra_points];
X = X_base;

Xmin = min(X);
Xmax = max(X);
Xnorm = 2*((X - Xmin)./(Xmax - Xmin)) - 1;

Yd = [2*X(:,1)+X(:,2).^2, X(:,1).^2+2*X(:,2)];

epoch = 100;
Nh = 15;
E = 1;
eta = 0.01;
momentum = 0.7;
Cost_epoch = [];

limit1 = sqrt(6/(2+Nh));
W1 = rand(Nh,2)*2*limit1 - limit1;
bias1 = rand(Nh,1)*2*limit1 - limit1;

limit2 = sqrt(6/(Nh+2));
W2 = rand(2,Nh)*2*limit2 - limit2;
bias2 = rand(2,1)*2*limit2 - limit2;

delta_W1_prev = zeros(size(W1));
delta_bias1_prev = zeros(size(bias1));
delta_W2_prev = zeros(size(W2));
delta_bias2_prev = zeros(size(bias2));

for iterasyon = 1:epoch
    for i = 1:size(X,1)
        x_input = Xnorm(i,:)';
        target = Yd(i,:)';

        Net1 = W1 * x_input + E * bias1;
        Y1 = logsig(Net1);

        Net2 = W2 * Y1 + E * bias2;
        Y2 = Net2; % PURELIN aktivasyon

        e = target - Y2;
        Cost(i) = 0.5 * (e' * e);

        yerel_gradyen2 = e; % purelin türevi = 1
        A = (yerel_gradyen2' * W2)';
        yerel_gradyen1 = A .* (Y1 .* (1 - Y1));

        delta_W2 = eta * yerel_gradyen2 * Y1' + momentum * delta_W2_prev;
        W2 = W2 + delta_W2;
        delta_W2_prev = delta_W2;

        delta_bias2 = eta * yerel_gradyen2 + momentum * delta_bias2_prev;
        bias2 = bias2 + delta_bias2;
        delta_bias2_prev = delta_bias2;

        delta_W1 = eta * yerel_gradyen1 * x_input' + momentum * delta_W1_prev;
        W1 = W1 + delta_W1;
        delta_W1_prev = delta_W1;

        delta_bias1 = eta * yerel_gradyen1 + momentum * delta_bias1_prev;
        bias1 = bias1 + delta_bias1;
        delta_bias1_prev = delta_bias1;
    end
    Cost_epoch = [Cost_epoch sum(Cost)];
    eta = eta * 0.995;

    figure(2);
    plot(Cost_epoch, 'r-');
    title('Eğitim'), xlabel('Epoch'), ylabel('Hata (Cost)');
    drawnow
end
toc

x1 = -5:0.5:5;
x2 = 5:-0.5:-5;
[X1, X2] = meshgrid(x1, x2);

Yd1 = 2.*X1 + X2.^2;
Yd2 = X1.^2 + 2.*X2;

figure(3);
subplot(2,2,1), mesh(X1,X2,Yd1), title('Çıkış1 istenen yüzey');
hold on, plot3(X(:,1),X(:,2),Yd(:,1),'rd');

subplot(2,2,2), mesh(X1,X2,Yd2), title('Çıkış2 istenen yüzey');
hold on, plot3(X(:,1),X(:,2),Yd(:,2),'rd');

for m = 1:size(X1,1)
    for n = 1:size(X1,2)
        x_norm = [X1(m,n), X2(m,n)];
        x_norm = 2*((x_norm - Xmin)./(Xmax - Xmin)) - 1;

        Net1 = W1 * x_norm' + E * bias1;
        Y1 = logsig(Net1);
        Net2 = W2 * Y1 + E * bias2;
        Y2 = Net2;
        Ag_cikis1(m,n) = Y2(1);
        Ag_cikis2(m,n) = Y2(2);
    end
end

subplot(2,2,3), mesh(X1,X2,Ag_cikis1), title('Çıkış1 elde edilen yüzey');
hold on, plot3(X(:,1),X(:,2),Yd(:,1),'rd');

subplot(2,2,4), mesh(X1,X2,Ag_cikis2), title('Çıkış2 elde edilen yüzey');
hold on, plot3(X(:,1),X(:,2),Yd(:,2),'rd');

X_test = [-3 -3; 3 3; 0 4; 4 0; -4 0; 0 -4; 1 1; -1 -1; 2 -2; -2 2];
Y_test = [2*X_test(:,1)+X_test(:,2).^2, X_test(:,1).^2+2*X_test(:,2)];

X_test_norm = 2*((X_test - Xmin)./(Xmax - Xmin)) - 1;

Y_pred = zeros(size(Y_test));
for i = 1:size(X_test,1)
    x_input = X_test_norm(i,:)';
    Net1 = W1 * x_input + E * bias1;
    Y1 = logsig(Net1);
    Net2 = W2 * Y1 + E * bias2;
    Y_pred(i,:) = Net2';
end

mse = mean((Y_test - Y_pred).^2);

max_range = 60; % yaklaşık aralık
perf1 = 1 - mse(1)/max_range^2;
perf2 = 1 - mse(2)/max_range^2;

sonuc = sprintf(['Test MSE (Çıkış 1): %.4f\n', ...
                 'Test MSE (Çıkış 2): %.4f\n', ...
                 'Performans (Çıkış 1): %.2f\n', ...
                 'Performans (Çıkış 2): %.2f\n'], ...
                 mse(1), mse(2), perf1, perf2);
disp(sonuc);

more off;
