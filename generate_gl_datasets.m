% This code generates the datasets for the examples written in the folder
% "examples".
% To generate all the datasets run the command
% generate_gl_datasets()

function generate_gl_datasets()
    % Add examples to the MATLAB path
    addpath('examples')

    % List all files in examples
    list_examples = struct2cell(dir('examples'));
    for i = 1:length(list_examples)
        example_name = list_examples{1,i};
        if strcmp(example_name(end),'m')
            example_name = example_name(1:end-2);
            generate_gl_example(example_name)
        end
    end
end