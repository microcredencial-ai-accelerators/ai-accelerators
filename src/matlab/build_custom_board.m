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

% Create the processor configuration object
hPC = dlhdl.ProcessorConfig();

hPC.TargetPlatform = 'Terasic DE10-Nano SoC';

% Add the synthesis tool to the MATLAB path. Uncomment the line below and
% provide the correct path.
hdlsetuptoolpath('ToolName', 'Altera Quartus II','ToolPath', '/opt/FPGA/Intel/intelFPGA/21.1std/quartus/bin/quartus');


% Create workflow config object
hWC = hdlcoder.WorkflowConfig('SynthesisTool', hPC.SynthesisTool,'TargetWorkflow','Deep Learning Processor');
hWC.RunTaskGenerateRTLCodeAndIPCore = true;
hWC.RunTaskCreateProject = true;
hWC.RunTaskBuildFPGABitstream = true; % Changed to True

% Call dlhdl.buildProcessor. This generates HDL code and creates the
% Quartus project.
dlhdl.buildProcessor(hPC, 'WorkflowConfig', hWC, 'OverrideResourceCheck', true);