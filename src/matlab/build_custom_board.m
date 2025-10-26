% This script creates a dlhdl.ProcessorConfig object for target Intel Arria
% 10 SoC, and then calls dlhdl.buildProcessor to generate HDL code and IP
% Core, and create a Quartus project. It skips the final step of synthesis
% and bitstream generation.
%
% Required products and support packages to run this script.
% MATLAB
% HDL Coder
% Deep Learning HDL Toolbox
% Deep Learning HDL Toolbox Support Package for Intel FPGA and SoC Devices
% xterm
% https://es.mathworks.com/matlabcentral/answers/1939799-deep-learning-hdl-toolbox-de10-standard

% matlab -nodesktop -nosplash -nodesktop -r "build_custom_board"

addpath(genpath('boards'))
% Create the processor configuration object
hPC = dlhdl.ProcessorConfig();
hPC.setModuleProperty("conv","ConvThreadNumber",4)
hPC.setModuleProperty("conv","InputMemorySize",[28 28 1])
hPC.setModuleProperty("conv","OutputMemorySize",[10 10 1])
hPC.setModuleProperty("conv","FeatureSizeLimit", 1024)

hPC.setModuleProperty("fc","FCThreadNumber",4)
hPC.setModuleProperty("fc","InputMemorySize",1024)
hPC.setModuleProperty("fc","OutputMemorySize",1024)

hPC.ProcessorDataType='int8'
hPC.UseVendorLibrary ='off' % Option to use vendor-specific floating point libraries

hPC.TargetPlatform = 'Terasic DE10-Nano SoC';

% Add the synthesis tool to the MATLAB path. Uncomment the line below and
% provide the correct path.
% quartus_path = strcat(getenv("QUARTUS_ROOTDIR"), "/bin/quartus");
quartus_path = '/opt/FPGA/Intel/intelFPGA/21.1std/quartus/bin/quartus'

hdlsetuptoolpath('ToolName', 'Altera Quartus II','ToolPath', quartus_path);


% Create workflow config object
hWC = hdlcoder.WorkflowConfig('SynthesisTool', hPC.SynthesisTool,'TargetWorkflow','Deep Learning Processor');
hWC.RunTaskGenerateRTLCodeAndIPCore = true;
hWC.RunTaskCreateProject = true;
hWC.RunTaskBuildFPGABitstream = true; % Changed to True

% Call dlhdl.buildProcessor. This generates HDL code and creates the
% Quartus project.
dlhdl.buildProcessor(hPC, 'WorkflowConfig', hWC, 'OverrideResourceCheck', true);