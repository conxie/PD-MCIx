import gradio as gr
import pandas as pd
import tempfile
from autoDX import PD_MCIx

test_type_options = ["Cutoff", "Series", "Already binary in data"]
DOMAINS = ['Attention', 'Language', 'Executive Function', 'Memory', 'Visuospatial']
TEST_NAMES = [f"{domain} Test {i}" for domain in DOMAINS for i in range(1, 3)]

NUM_NP_TESTS = 10
NP_TEST_INPUT_COUNT = 2
NUM_FUNC_TESTS = 4
NUM_SUBJ_TESTS = 4

objective_inputs = []
subjective_inputs = []
functional_inputs = []

def save_tests(*args):
    if not args[-3]:
        raise gr.Error('Must upload CSV before running')
    df = pd.read_csv(args[-3].name)
    df_columns = df.columns
    missing = []

    NP_TEST_STOP = NUM_NP_TESTS * NP_TEST_INPUT_COUNT
    # create structure for Neuropsychological assessment
    np_tests = []
    j=0
    for i in range(0, NP_TEST_STOP, NP_TEST_INPUT_COUNT):
        name = args[i]
        #if name=='':
        #    raise gr.Error(f'Missing column name for {TEST_NAMES[j]}')
        if name not in df_columns:
            missing.append(name)
        #if cutoff == '':
            #raise gr.Error(f'Missing cutoff for {TEST_NAMES[j]}')
        cutoff = args[i+1]
        np_tests.append((name, cutoff))
        j+=1
    
    # create structure for functional/subjective tests
    FUNCTIONAL_COUNT = int(args[-1])
    SUBJECTIVE_COUNT = int(args[-2])
    SUBJECTIVE_TEST_STOP = NP_TEST_STOP + 3*NUM_SUBJ_TESTS
    subjective_tests = {}
    j=1
    for i in range(NP_TEST_STOP, SUBJECTIVE_TEST_STOP, 3):
        if j>SUBJECTIVE_COUNT:
            continue
        if args[i] not in df_columns:
            missing.append(args[i])
        subjective_tests[j] = {'col': args[i], 'type': args[i+1], 'val': args[i+2]} 
        j+=1
    
    FUNCTIONAL_TEST_STOP = SUBJECTIVE_TEST_STOP + 3*NUM_FUNC_TESTS
    j=1
    functional_tests = {}
    for i in range(SUBJECTIVE_TEST_STOP, FUNCTIONAL_TEST_STOP, 3):
        if j>FUNCTIONAL_COUNT:
            continue
        if args[i] not in df_columns:
            missing.append(args[i])
        functional_tests[j] = {'col': args[i], 'type': args[i+1], 'val': args[i+2]}
        
    #if missing:
        #raise gr.Error(f"Missing columns in CSV: {', '.join(missing)}")
    
    #new_df = PD_MCIx(df, np_tests, subjective_tests, functional_tests)
    new_df = df
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as tmp:
        new_df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    return gr.update(visible=True), new_df.head(5), tmp_path, gr.update(visible=True)

def toggle_input(input_type):
    if input_type == "Cutoff":
        return gr.Textbox(visible=True, label="", placeholder="Cutoff")
    elif input_type == "Series":
        return gr.Textbox(visible=True, label="", placeholder="Comma-separated values")
    else:
        return gr.Textbox(visible=False, label="")


with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Test Configuration for Cognitive Domains")
    gr.Markdown("**Note:** You must enter *two tests* for each of the five domains listed below. "
                "Each test must include a name, an input type, and a cutoff.")

    for domain in DOMAINS:
        with gr.Accordion(domain, open=False):
            for t in range(1, 3):
                with gr.Row():
                    name = gr.Textbox(label='Test name in CSV', placeholder=f"{domain} Test {t} Name", scale=3)
                    value_input = gr.Textbox(label="Cutoff", placeholder="Cutoff", scale=3)
                    objective_inputs.extend([name, value_input])

    gr.Markdown("## 🧠 Required Subjective & Functional Impairment Tests")
    gr.Markdown("You must enter **at least one test** (1–4 allowed) for each category below.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Subjective Impairments")
            subjective_count = gr.Dropdown(choices=["1", "2", "3", "4"], label="Number of subjective tests", value="1")
            subjective_container = gr.Column(visible=True)
            subjective_rows = []
            for i in range(4):
                with subjective_container:
                    with gr.Row(visible=(i == 0)) as row:
                        name = gr.Textbox(label='Subjective test name in CSV', placeholder=f"Subjective Test {i+1}", scale=3)
                        test_type = gr.Dropdown(label='Criteria for impairment', choices=test_type_options, value="Cutoff", scale=2)
                        value_input = gr.Textbox(label="", placeholder="Cutoff", scale=3)
                        test_type.change(fn=toggle_input, inputs=test_type, outputs=value_input)
                        subjective_inputs.extend([name, test_type, value_input])
                    subjective_rows.append(row)

        with gr.Column():
            gr.Markdown("### Functional Impairments")
            functional_count = gr.Dropdown(choices=["1", "2", "3", "4"], label="Number of functional tests", value="1")
            functional_container = gr.Column(visible=True)
            functional_rows = []
            for i in range(4):
                with functional_container:
                    with gr.Row(visible=(i == 0)) as row:
                        name = gr.Textbox(label='Functional test name in CSV', placeholder=f"Functional Test {i+1}", scale=3)
                        test_type = gr.Dropdown(label='Criteria for impairment', choices=test_type_options, value="Cutoff", scale=2)
                        value_input = gr.Textbox(label="", placeholder="Cutoff", scale=3)
                        test_type.change(fn=toggle_input, inputs=test_type, outputs=value_input)
                        functional_inputs.extend([name, test_type, value_input])
                    functional_rows.append(row)

    # --- Visibility toggles ---
    def update_subjective_visibility(n):
        n = int(n)
        return [gr.update(visible=True)] + [gr.update(visible=i < n) for i in range(4)]

    def update_functional_visibility(n):
        n = int(n)
        return [gr.update(visible=True)] + [gr.update(visible=i < n) for i in range(4)]

    subjective_count.change(fn=update_subjective_visibility,
                            inputs=subjective_count,
                            outputs=[subjective_container] + subjective_rows)

    functional_count.change(fn=update_functional_visibility,
                            inputs=functional_count,
                            outputs=[functional_container] + functional_rows)

    # --- Validation and Submission ---
    gr.Markdown("## 📂 Upload Clinical CSV Data")
    gr.Markdown("Upload your patient-level clinical data (CSV format). The column names must match those used above.")

    csv_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
    csv_preview = gr.Dataframe(label="CSV Preview", visible=False)
    submit_btn = gr.Button("Submit", interactive=True)    

    
    output_df_preview = gr.Dataframe(label="Processed Output", visible=False)
    output_download = gr.File(label="Download Processed CSV", visible=False)

    def show_csv(file):
        try:
            df = pd.read_csv(file.name)
            return gr.update(visible=True), df.head()
        except Exception as e:
            return gr.update(visible=False), pd.DataFrame([["Error loading file:", str(e)]])

    csv_upload.change(fn=show_csv, inputs=csv_upload, outputs=[csv_preview, csv_preview])

    all_inputs = objective_inputs + subjective_inputs + functional_inputs

    submit_btn.click(fn=save_tests, inputs=all_inputs + [csv_upload] + [subjective_count] + [functional_count], outputs=[output_df_preview, output_df_preview, output_download, output_download])
    

demo.launch()
