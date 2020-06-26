function [ptseries] = ciftiParcellateGlasserFrom64k(datafile64k, outputfile)
%% Taku Ito
%% 03/27/2017
%% Parcellates 64k dtseries into 360 x time CSV file output file
%% Uses Glasser et al., 2016 atlas
%% Organized as L->R; Rows 1:180 = L; Rows 181:360 = R

    lparcels = '/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';
    rparcels = '/projects/AnalysisTools/ParcelsGlasser2016/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii';

    %lparcels = ciftiopen(lparcels,'wb_command');
    %rparcels = ciftiopen(rparcels,'wb_command');

    %data64k = ciftiopen(datafile64k,'wb_command);
    %numTime = size(data64k.cdata,2);
    %nParcels = 360;
    %ptseries = zeros(nParcels,numTime);
    L_parcelTSFilename = [outputfile '_L.ptseries.nii'];
    R_parcelTSFilename = [outputfile '_R.ptseries.nii'];

    eval(['!wb_command -cifti-parcellate ' datafile64k ' ' lparcels ' COLUMN ' L_parcelTSFilename ' -method MEAN'])
    eval(['!wb_command -cifti-parcellate ' datafile64k ' ' rparcels ' COLUMN ' R_parcelTSFilename ' -method MEAN'])

    lptseries = ciftiopen(L_parcelTSFilename,'wb_command');
    rptseries = ciftiopen(R_parcelTSFilename,'wb_command');

    ptseries = [lptseries.cdata; rptseries.cdata;];

    csvwrite([outputfile '_LR.csv'], ptseries);

end


