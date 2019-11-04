%run after PCAscript
img = 'phughe.16.jpg';   

A = imread(img);%read RGB channels

A = A(coord(1,1):coord(2,1),coord(1,2):coord(2,2),:);%extract subimages with useful information

D = (A(:))';%write the row

E = (double(D)-mu)*cff;%converts to PC space

class = classify(E(1:4),score(:,1:4),T, 'linear' );%classify

%gscatter(E(1),E(2),class,'rbcy','x',5,'off');
plot(E(1),E(2), 'x' ,'MarkerSize', 9 ,'LineWidth',1, ...
    'MarkerFaceColor','k' , 'MarkerEdgeColor','k'      );
text( E(1),E(2), {'  '; ['Clase:' int2str(class)]} ,'FontSize', 8 );