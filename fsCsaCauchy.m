% A hybrid Artificial Immune optimization for high-dimensional feature selection
% ------------- for HFSIA-------------
% Yongbin Zhu, 2021-01-05
%%
function Csa = fsCsaCauchy(feat,label,opts)
    N=10;
    max_Iter=50;
    CR = 0.5;
    d = -0.2;
    
    if isfield(opts,'N'), N = opts.N; end
    if isfield(opts,'T'), max_Iter = opts.T; end
    if isfield(opts,'CR'), CR = opts.CR; end
    
    dim = size(feat,2);
    Pop = newPopCauchy(N,dim,d);
    t=1;
    clN = round(N * CR);
    if (clN == 0), clN = 1; end

    fitCurve = zeros(1,max_Iter); 
    pt=max_Iter;

    while t<=max_Iter
        fitP = FitnessFunction(feat,label,Pop,opts);  
        [stFitP,idx]=sort(fitP,'ascend');
        rw= size(Pop,1);
        if rw > N
            slP = Pop(idx(1:N),:);
        else
            slP = Pop;
        end
        
        fitCurve(t) = stFitP(1);
        bestAb = slP(1,:);
        
        if rw < clN
            tpC = Pop;
        else
            tpC = slP(1:clN,:);
        end

        if t > 3
            flg = (fitCurve(t)< fitCurve(t-1)) & (fitCurve(t-1) < fitCurve(t-2));
            if flg && (d > -2)
                d = -round(t/pt,3) + 0.5;
            end
        end

        c = [];
        Np = size(tpC,1);
        j = Np;
        for i = 1:Np
            k = tpC(i,:);
            b = repmat(k,j,1);
            c = [c;b];
            j = j-1;
        end
     
        [m,n]=size(c);
        a = newPopCauchy(m,n,d);
        mtC = c - a;
        mtC(mtC<1)=0;
        
        dd = round((t/pt),3);
        newP = newPopCauchy(N,dim,dd);
        tpD = [bestAb;mtC;newP];
        newD = unique(tpD,'rows','stable');
        Pop = newD;
        fprintf('Generation %d Best (CSA)= %f \n',t,fitCurve(t))
        t = t +1;
    end
    
    [~,Sf] = find(bestAb == 1);
    slFeat = feat(:,Sf);
    Csa.sf = Sf;
    Csa.ff = slFeat;
    Csa.c = fitCurve;
end

%%
function newp = newPopCauchy(N,dim,d)
    orix=rand(N,dim);
    cx=tan((orix-1/2)*pi);
    cx(cx>=d) = 1;
    cx(cx<d) = 0;
    newp = cx;
end

%%
function fit = FitnessFunction(feat,label,solution,opts)
    len = size(solution,1);
    fit = zeros(1,len);
    ws = [0.99; 0.01];
    alpha = ws(1); 
    beta = ws(2);
    for i = 1:len
        X = solution(i,:);
        lag = sum(X == 1);
        if lag == 0
          cost = 1;
        else
          sFeat = feat(:,X == 1);
          error    = subsetEvaKNN(sFeat,label,opts);
          num_feat = sum(X == 1);
          max_feat = length(X);
          cost = alpha * error + beta * (num_feat / max_feat); 
        end
        fit(i) = cost;
    end
end

%%
function error = subsetEvaKNN(sFeat,label,opts)
    if isfield(opts,'k'), k = opts.k; end
    if isfield(opts,'Model'), Model = opts.Model; end

    trainIdx = Model.training;    testIdx = Model.test;
    xtrain   = sFeat(trainIdx,:); ytrain  = label(trainIdx);
    xvalid   = sFeat(testIdx,:);  yvalid  = label(testIdx);
    My_Model = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
    pred     = predict(My_Model,xvalid);
    Acc      = sum(pred == yvalid) / length(yvalid);
    error    = 1 - Acc;
end