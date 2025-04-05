function [rank] = fsFisher(feat,label)
    numC = max(label);
    [~, numF] = size(feat);
    out.W = zeros(1,numF);

    cIDX = cell(numC,1);
    n_i = zeros(numC,1);
    for j = 1:numC
        cIDX{j} = find(label(:)==j);
        n_i(j) = length(cIDX{j});
    end

    for i = 1:numF
        temp1 = 0;
        temp2 = 0;
        f_i = feat(:,i);
        u_i = mean(f_i);

        for j = 1:numC
            u_cj = mean(f_i(cIDX{j}));
            var_cj = var(f_i(cIDX{j}),1);
            temp1 = temp1 + n_i(j) * (u_cj-u_i)^2;
            temp2 = temp2 + n_i(j) * var_cj;
        end

        if temp1 == 0
            out.W(i) = 0;
        else
            if temp2 == 0
                out.W(i) = 100;
            else
                out.W(i) = temp1/temp2;
            end
        end
    end
[~, rank] = sort(out.W, 'descend');

