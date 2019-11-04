function [coeff,score,latent,explained,mu,y] = pcag(x)%PCA using the Gram matrix eigenvalues as described by Turk and Pentland in their eigenfaces paper

if nargin > 1
    error('Too many arguments/n');
end

mu=mean(x);%needed to center future data for classification purposes

x=bsxfun(@minus, x, mu);%centering the data

gram=x*x';

[coeff,latent]=eig(gram);

%[coeff,latent]=sortem(coeff,latent);

coeff=(x'*coeff)/sqrt(abs(latent));%this normalization method is not very accurate if eigenvalues are near zero
%coeff=(x'*coeff);
%n = sqrt(sum(coeff.^2,1)); % Compute norms of columns
%coeff = bsxfun(@rdivide,coeff,n); % Normalize M (even less accurate)
   
latent=diag(latent)/(size(x,1)-1);%each number indicates how much effect it has on the variance
                                    %adjusted so they are the covariance matrix eigenvalues 
                                    
[latent,index]=sort(latent,'descend');%order eigenvalues by descendent order

coeff=coeff(:,index);%order eigenvectors

y=latent;%y is a debug variable

explained=100*(latent/sum(latent));%percentage of variance explained by a principal component

%coeff=normc(coeff);%each column is a principal component

score=x*coeff;%each row is a vector representation in the PC space