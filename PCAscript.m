clear;

tic;

S = char('kaknig.','phughe.','vstros.');

filenames=0;
filespergroup=4;
files2read = size(S,1)*filespergroup;%number of files to read from the string array

coord = [1 1;200 180];%[70 70;469 469];%subimage extreme points' coordinates

for xx=1:size(S,1)
    for yy=1:filespergroup
        filenames = char(filenames,sprintf('%s%d.jpg',S(xx,:),yy));%string array with image file names
    end
end

filenames=filenames(2:(files2read+1),:);

v = (coord(2,1)-coord(1,1)+1)*(coord(2,2)-coord(1,2)+1);%480*640;%number of pixels

fig_num = 1;

titulo = 'PCA - Clasificador lineal (con más detalles)';

IM = zeros(files2read,v*3);%memory preallocation for the image vector array

for i = 1:files2read
    
    img = filenames(i,:);   

    A = imread(img);%read RGB channels
    
    A = A(coord(1,1):coord(2,1),coord(1,2):coord(2,2),:);%extract subimages with useful information

    IM(i,:) = (A(:))';%write the row

end

[cff,score,latent,explained,mu,y] = pcag(IM);%does a pca using the Gram matrix to save memory
clear IM;

[f,c] = size (score);%f is the number of vectors, c is the number of variables 

figure (fig_num)
 %   clf (fig_num);
title( titulo,'FontSize',10) ;
hold on ;
        
[T_real,cnames] = grp2idx(filenames(:,1));%groups images using the first character of the filename

T = T_real(1:files2read) ;
  
ms =  [ 'o' ; '^' ; 's' ; 'v' ; '>' ; 'o' ; '<' ; 's' ;   'o' ; '^' ; 'o' ; '^' ; 's' ; 'v' ; '>' ; 'p' ; '<' ; 'h' ; 'o' ; '^' ;'o' ; '^' ; 's' ; 'v' ; '>' ; 'p' ; '<' ; 'h' ; 'o' ; '^' ];  % Marker Specifiers   

mfc = [  'b'  ; 'y' ; 'w' ; 'g' ; 'c' ; 'k' ; 'm' ; 'w'; 'w'; 'w';  'r' ; 'y' ; 'b'; 'g'; 'c'; 'k'; 'm'; 'w'; 'w'; 'w';  'r' ; 'y' ; 'b'; 'g'; 'c'; 'k'; 'm'; 'w'; 'w'; 'w' ];  % MarkerFaceColor    


mec = [ 'k' ; 'k' ; 'k'; 'k'; 'k' ; 'k' ; 'k'; 'k'; 'k'; 'k' ;  'k' ; 'k' ; 'k'; 'k'; 'k' ; 'k' ; 'k'; 'k'; 'k'; 'k' ; 'k' ; 'k' ; 'k'; 'k'; 'k' ; 'k' ; 'k'; 'k'; 'k'; 'k' ];  %  'MarkerEdgeColor','k' es negro  
for jj=1 : max(T)        
    plot(score(find(T==jj),1),score(find(T==jj),2), ms(jj) ,'MarkerSize', 9 ,'LineWidth',1, ...
    'MarkerFaceColor',mfc(jj) , 'MarkerEdgeColor',mec(jj)      ); %fancy plot     
end

for i=1:f
    text( score(i,1),score(i,2),{'  '; ['  ' filenames(i,:)] } ,'FontSize', 8 );
end

grid on ;

box on ;

%classifier
ss=get(fig_num); 
axis_act = [ get(ss.CurrentAxes,'XLim') get(ss.CurrentAxes,'YLim') ] ;
f= 1.3 ; %zoom factor
xlim([axis_act(1)*f axis_act(2)*f]) ;
ylim([axis_act(3)*f axis_act(4)*f]) ;
axis_act = [ get(ss.CurrentAxes,'XLim') get(ss.CurrentAxes,'YLim') ] ;
% Classify a grid of measurements on the same scale:   
[X,Y] = meshgrid(linspace(axis_act(1),axis_act(2)),linspace(axis_act(3),axis_act(4)) );%creates 2D mesh
X = X(:); Y = Y(:);%convert the mesh arrays to vectors
[class,err,P,logp,coeff] = classify([X Y],score(:,1:2),T, 'linear' );%applies a linear classifier to the mesh points
%Visualize the classification:
gscatter(X,Y,class,'rbcy','.',1,'off');

%for gg=1 : (max(T)-1) 
%    K = coeff(gg,1+gg).const; L = coeff(gg,1+gg).linear; 
%    fxy = sprintf('0 = %g+%g*x+%g*y', K,L(1),L(2));        
%    h = ezplot(fxy,axis_act);     set(h,'Color','k','LineWidth',1)  %'k':black m'
%end %WARNING: it may not generalize properly to a higher number of groups
 
s1=['Primera Componente Principal ' int2str(explained(1)) ' %'];
s2=['Segunda Componente Principal ' int2str(explained(2)) ' %'];
xlabel(s1)  ;   ylabel(s2)  ;
 
toc;