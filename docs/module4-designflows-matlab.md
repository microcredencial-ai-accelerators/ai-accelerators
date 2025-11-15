# Creating a MATLAB Deep Learning Processor for the Terasic DE10-Nano SoC
## [Back to Module 4](module4-designflows.md)

This tutorial guides you through creating a **Deep Learning Processor (DL Processor)** for the **Terasic DE10-Nano SoC** using **MATLAB**, **HDL Coder**, and **Deep Learning HDL Toolbox**.

---

## Prerequisites

Before you start, ensure that the following software and packages are **installed and licensed**:

### Required Products
- **MATLAB**
- **[HDL Coder](https://mathworks.com/products/hdl-coder.html)**
- **[Deep Learning Toolbox](https://mathworks.com/products/deep-learning.html)**
- **[Deep Learning HDL Toolbox](https://mathworks.com/products/deep-learning-hdl.html)**
- **[Deep Learning Toolbox Converter for TensorFlow Models](https://mathworks.com/matlabcentral/fileexchange/64649-deep-learning-toolbox-converter-for-tensorflow-models)**
- **[Deep Learning HDL Toolbox Support Package for Intel FPGA and SoC Devices](https://es.mathworks.com/hardware-support/deep-learning-intel-fpga.html)**
- **xterm** (for Linux users)

> MATLAB version used: **R2025b**  
> Target platform: **Intel® Arria® 10 / Terasic DE10-Nano SoC**

---

## Project Structure

Organize your working directory as follows:
```
project_root/
├── build_custom_board.m % Main script
├── saved_models/
│ └── mnist_fc/
│ └── model.h5 % Trained Keras model
└── boards/
├── +DE10Nano/
│ ├── +dl_soc_reference/
│ │ ├── plugin_rd.m
│ │ ├── system_soc.qsys
│ │ └── system_soc.tcl
│ ├── hdlcoder_ref_design_customization.m
│ └── plugin_board.m
└── hdlcoder_board_customization.m
```
The board definition tells MATLAB how to interface with your DE10-Nano design. You need to define board definition and interfaces according to [Deep Learning Processor IP Core Generation for Custom Board](https://es.mathworks.com/help/deep-learning-hdl/ug/define-custom-board-and-reference-design-for-dl-ip-core-workflow.html)

## Step 1. Load and Prepare the Neural Network

We’ll import a trained **Keras model** (`.h5` file) and prepare it for HDL conversion.

```matlab
model_path = 'saved_models/mnist_fc/model.h5';
net = importKerasNetwork(model_path);
info = analyzeNetwork(net);

% Update the classification output layer
correctClasses = string(0:9);
layers = net.Layers;
newClassificationLayer = classificationLayer('Classes', categorical(correctClasses), 'Name', 'output');
layers(end) = newClassificationLayer;

% Reassemble and analyze the updated network
net = assembleNetwork(layers);
info = analyzeNetwork(net);
```

## Step 2. Create the Processor Configuration
Create and configure a dlhdl.ProcessorConfig object that defines the DL processor architecture.
```matlab
hPC = dlhdl.ProcessorConfig();

% Convolution module
hPC.setModuleProperty("conv", "ConvThreadNumber", 4);
hPC.setModuleProperty("conv", "InputMemorySize", [28 28 1]);
hPC.setModuleProperty("conv", "OutputMemorySize", [10 10 1]);
hPC.setModuleProperty("conv", "FeatureSizeLimit", 1024);

% Fully connected (FC) module
hPC.setModuleProperty("fc", "FCThreadNumber", 4);
hPC.setModuleProperty("fc", "InputMemorySize", 1024);
hPC.setModuleProperty("fc", "OutputMemorySize", 1024);
```

Reource usage can be estimated with:
```matlab
 hPC.estimateResources;
```
Example result for Intel Arria 10 SoC development kit with single precision:
```
Deep Learning Processor Estimator Resource Results

                            DSPs          Block RAM*     LUTs(CLB/ALUT)
                        -------------    -------------    ------------- 
DL_Processor                 231           16442624          117943
* Block RAM represents Block RAM tiles in Xilinx devices and Block RAM bits in Intel devices
```

### Optional Optimization
Automatically tune the processor parameters for your network:
```matlab
hPC.optimizeConfigurationForNetwork(net);
```
by default dlproccesor precision is set to 'single'  You can configure your processor for 'int8' precision:
```
hPC.ProcessorDataType = 'int8';
hPC.UseVendorLibrary = 'off';
```
## Step 3. Define the Target Platform
```matlab
PC.TargetPlatform = 'Terasic DE10-Nano SoC';

% Specify the Quartus installation path
quartus_path = '/opt/FPGA/Intel/intelFPGA/21.1std/quartus/bin/quartus';
hdlsetuptoolpath('ToolName', 'Altera Quartus II', 'ToolPath', quartus_path);
```
## Step 4. Configure the HDL Workflow
The WorkflowConfig object defines which steps to execute during processor generation.
```matlab
hWC = hdlcoder.WorkflowConfig('SynthesisTool', hPC.SynthesisTool, ...
                              'TargetWorkflow', 'Deep Learning Processor');

hWC.RunTaskGenerateRTLCodeAndIPCore = true;
hWC.RunTaskCreateProject = true;
hWC.RunTaskBuildFPGABitstream = true;  % set to false to skip synthesis
```
## Step 5. Build the DL Processor
Generate HDL code, IP Core, and the Quartus project.
```matlab
dlhdl.buildProcessor(hPC, 'WorkflowConfig', hWC, 'OverrideResourceCheck', true);
```
This command:

1. Generates HDL code from your processor configuration.
2. Creates a Quartus project for synthesis.
2. Builds the FPGA bitstream (if enabled).

## Step 6. Bitstream
'.bin' file will be generated with dlprocessor bitstream.
