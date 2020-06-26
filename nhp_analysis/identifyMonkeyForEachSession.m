% Taku Ito
% Analyzing Siegel et al. 2015 data set 
% 08/10/2018

%clear all; close all;
datadir = '/projects3/NHPActFlow/data/';
sessionNames = {'100706','100730','100804','100820','100827','100913','100921','101024','101122','101128','101207','101217','110110_01','110115_01','100724','100731','100817','100823','100828','100915','101008','101027','101123','101202','101209','110106','110110_02','110120','100725','100802','100818','100824','100907','100917','101009','101028','101124','101203','101210','110107_01','110111_01','110121','100726','100803','100819','100826','100910','100920','101023','101030','101127','101206','101216','110107_02','110111_02.mat'};

datastruct.session = {};
datastruct.name = {};

for i=1:length(sessionNames)
    datadir = '/projects3/NHPActFlow/data/';
    
    %% First load in session
    disp(['Loading session ' sessionNames{i}])
    disp(['Session ' num2str(i) ' out of ' num2str(length(sessionNames))])
    load([datadir sessionNames{i}]);
    close all
    clear ain
    clear lfp
    clear lfpSchema
    clear spikeTimes
    clear ainSchema

    if ~isempty(regexp(fileInfo.file{1},'paula'))
        name = 'paula';
        disp(['Session ' sessionNames{i} ' is paula'])
    end
    if ~isempty(regexp(fileInfo.file{1},'rex'))
        name = 'rex';
        disp(['Session ' sessionNames{i} ' is rex'])
    end

    datastruct.session{i} = sessionNames{i};
    datastruct.name{i} = name;

end

% Save task data
tmptable = struct2table(datastruct);
writetable(tmptable,['/projects3/TaskFCMech/data/nhpData/monkeyToSessionID.csv']);

    
