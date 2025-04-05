% A hybrid Artificial Immune optimization for high-dimensional feature selection
% ------------- for HFSIA-------------
% Yongbin Zhu, 2021-01-05
%%
clear, clc, close,clear all;

opts.k = 5;
opts.N  = 10;           % Population size
opts.T  = 50;           % maximum number of iterations
opts.CR = 0.5;          % select rate
opts.nf = 300;          % Fisher filter threshold
ho = 0.2;               % Ratio of validation data

filepath ='data';           %  ----- Data folder
matName = 'GLI-85.mat';      %  ----- Data file name

%% The main
matPath =strcat(filepath,'\',matName);
tempData = load(char(matPath));
val_names = fieldnames(tempData);
ftName = string(val_names{1});
lbName = string(val_names{2});

feat = double(tempData.(ftName));
label = tempData.(lbName);

if size(label,1) == 1, label=transpose(label);end
tempRlt = solve(feat,label,opts,ho);

%%
function slRlt = solve(feat,label,opts,ho)
    HO = cvpartition(label,'HoldOut',ho); 
    opts.Model = HO;

    % ===== To get subFeat by Fisher Score =====
    nFish = opts.nf;
    fun = @fsFisher;
    rkFhData = fun(feat,label);
    fstSubData = rkFhData(:,1:nFish);
    subFeat = feat(:,fstSubData);
    
    % ===== To get subsubFeat by fsCsaCauchy =====
    FsFun  =  @fsCsaCauchy;
    FS   =  FsFun(subFeat,label,opts);
    
    sf_idx = FS.sf;
    nf = length(sf_idx);
    sf_idx = FS.sf;
    
    fprintf('\n');
    nfStr=strcat('FeatNumber: ',num2str(nf));
    slFeatStr=strcat('selectFeat: ',num2str(sf_idx));
    
    disp(nfStr);
    disp(slFeatStr);
    
    slRlt.c = FS.c;
    slRlt.ff = FS.ff;
    slRlt.sf = FS.sf;
end
