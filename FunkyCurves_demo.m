function accuracy = FunkyCurves_demo()
% demo for SPA classifier on Funky Curves data set

%% output:
%        accuracies for Funky Curves data set with differen training
%        sample size


%% History:
%   Didong Li       September 10, 2018, created
%   Didong Li       March 2, 2019 modified
 

% samples from three funky curves with Gaussian noise
load('FunkyCurves_noise.mat')
data = Funkycurves_noise;
[n,p] = size(data);  % total sample size is 1500


X = data(:,1:p-1);
y = data(:,p);

% visualize the data set
figure
hold on
for i = 1:3
    plot(X(y==(i-1),1),X(y==(i-1),2),'*')
end
hold off
alpha = 0.2;

% hold 1200 for test data
data_test = data(floor(n*alpha+1):n,:);
X_te = data_test(:,1:p-1);
y_te = data_test(:,p);

% use the first 150, 180, 210, 240, 270, 300 samples for training
ns = n*[0.1:0.02:alpha];
nn = size(ns,2);

accuracy_spherelets = zeros(nn,1);

for ni= 1: nn
    data_ni = data(1:ns(1,ni),:);
    X_tr = data_ni(:,1:p-1);
    y_tr = data_ni(:,p);
    [k, d, accuracy] = SPA_tune(X_tr, y_tr); % tune k and d
    [label_te, accuracy_spherelets(ni,1)] = cls_spherelets(X_tr, y_tr, X_te, y_te, k, d);
    display(['Spherelets: k=',num2str(k), ', d=', num2str(d), ', n_tr=', num2str(ns(1,ni)), ', accuracy=', num2str(accuracy_spherelets(ni,1))])
end

% plot results
figure
plot(ns,accuracy_spherelets,'*-')
legend('spherelets')
xlabel('training sample size')
ylabel('accuracy')
return