load('seadepth1.mat');
load('seadepth2.mat');
load('seadepth3.mat');
load('seadepth4.mat');
seadepth=[seadepth1;seadepth2;seadepth3;seadepth4];
save('seadepth.mat','seadepth');

ProcessData;
ProcessDataFewerInducingPoints;
ProcessDataFewerMeasurements;