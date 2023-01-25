function [neg_devs, pos_devs]=negposdevs(Data)
    params=size(Data,2);
    pos_devs=zeros(params,1);
    neg_devs=zeros(params,1);
    for param=1:params
        mean_here=mean(Data(:,param));
        PositiveIndices=Data(:,param)>mean_here;
        NegativeIndices=Data(:,param)<=mean_here;
        pos_devs_all=Data(PositiveIndices,param);
        neg_devs_all=Data(NegativeIndices,param);
        pos_devs(param)=mean(pos_devs_all)-mean_here;
        neg_devs(param)=mean_here-mean(neg_devs_all);
    end
end