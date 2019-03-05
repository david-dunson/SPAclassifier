function [k, d, accuracy] = SPA_tune(X, y)
% Tune parameters for SPherical Approximation (SPA) classifier

%% input: 
%           X: training data
%           y: label of training data

%% Output:
%           k: the optimal number of neighbors
%           d: the optimal dimension of manifold
%           accuracy: the classification accuracy given above k and d

%% History:
%   Didong Li       March 2, 2019, created

%% 
[n,p] = size(X); % p is ambient dimension
Nfold = 5; % 5-fold cross validation
onefoldsize = floor(n/Nfold);

% make labels start from 1 instead of 0
if min(y)==0
    y = y+1;
end
N = max(y); % N is number of classes

% default choices for d. User may replace 20 by a larger integers if the intrinsic dimension is larger
d_max = min(20,p-1); 
ds = [1:1:d_max];
nd = length(ds); % number of choices for d

data = cell(N,1); % collect data in each class
n_cls = zeros(N,1);
for j =1:N
    data{j} = X(find(y==j),:);
    n_cls(j) = size(data{j},1); % sample size of class j
end
n_min = min(n_cls); % find the size of the smallest class

% default choices for k, ranging from 3 to n_min/2.
nk  = min(20,ceil(n_min/2-2)); % number of choices for k
step_size = max(floor((n_min/2-3)/nk),1); % difference between two choices, user may change this step size
ks = [3:step_size:ceil(n_min/2)];

% accuracy for each pair of d,k
accuracy_kd = zeros(nk,nd);


for j = 1:Nfold
    % split data into Nfold folds and use the j-th fold as validation set
    X_tr = X;
    y_tr = y;
    X_tr(onefoldsize*(j-1)+1:onefoldsize*j,:)=[];
    y_tr(onefoldsize*(j-1)+1:onefoldsize*j,:)=[];
    X_cv = X(onefoldsize*(j-1)+1:onefoldsize*j,:);
    y_cv = y(onefoldsize*(j-1)+1:onefoldsize*j,:);          
    for i = 1:nk
        for k = 1:nd
            if ks(1,i) > ds(1,k)+1
                [label_cv, curr_accuracy] = cls_spherelets(X_tr, y_tr, X_cv, y_cv, ks(1,i), ds(1,k));
                accuracy_kd(i,k) = accuracy_kd(i,k)+curr_accuracy;      
            else
                display('k<= d+1, no SPCA degenerates')
            end
        end
    end
end             
accuracy_kd = accuracy_kd/Nfold; %average the accuracy over folds
% find the pair of k and d with the highest accuracy
[accuracy, max_idx] = max(accuracy_kd(:));
[ind1, ind2]=ind2sub(size(accuracy_kd),max_idx);
k = ks(1, ind1); % optimal k
d = ds(1, ind2); % optimal d
return

