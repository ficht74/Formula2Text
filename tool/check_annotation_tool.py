import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import io
import json
import os

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("f(c) Check Annotations Tool")
        #master.geometry("10100x920")
        self.corrected_data = []

        # Set the default font for the whole GUI
        default_font = ("Arial", 16)
        master.option_add("*Font", default_font)

        # create upper frame
        self.upper_frame = tk.Frame(master, width=1000, height=200,bd=1, relief=tk.GROOVE)
        self.upper_frame.pack_propagate(False)
        self.upper_frame.pack(side="top")

        # create image label with default size of 300x300
        self.image_label = tk.Label(self.upper_frame, text="Loaded Image")
        self.image_label.grid(column=0, row=0, padx=5, pady=1)

        self.image_label_preview = tk.Label(self.upper_frame, text="LaTeX Preview")
        self.image_label_preview.grid(column=1, row=0, padx=5, pady=1)

        
        # ---------------------------------------------------
        # create lower frame
        self.lower_frame = tk.Frame(master, width=1000, height=300,bd=1, relief=tk.GROOVE)
        self.lower_frame.pack_propagate(False)
        self.lower_frame.pack(side="bottom")

        # create label and textfield "filename"

        self.filename_field_label = tk.Label(self.lower_frame, text="Image", font=('Arial', 18, "bold"),bd=1, relief="groove", padx=40, pady=7, background="lightblue")
        self.filename_field_label.grid(column=0, row=0,sticky="ew", columnspan=3)

        self.filename_field_label = tk.Label(self.lower_frame, text="Filename Image:")
        self.filename_field_label.grid(column=0, row=1, padx=5, pady=20)
        self.filename_field = tk.Entry(self.lower_frame,width=100)
        self.filename_field.grid(column=1, row=1, padx=5, pady=5)

        # create "Generate Latex" button
        self.generate_latex_button = tk.Button(self.lower_frame, text="Save & Next Entry", font=("Arial",14,"bold"), command=self.next_entry_save, foreground="magenta")
        self.generate_latex_button.grid(column=2, row=1, padx=5, pady=5)

        # ------------------------------------------------------------------

        self.latex_label = tk.Label(self.lower_frame, text="LaTeX-Code",  font=('Arial', 18, "bold"),bd=1, relief="groove", padx=40, pady=7, background="lightblue")
        self.latex_label.grid(column=0, row=2, sticky="ew", columnspan=3)
        
        # create label and textfield for "LaTeX Code Generated"
        self.generated_latex_label = tk.Label(self.lower_frame, text="LaTeX Generated:")
        self.generated_latex_label.grid(column=0, row=3, padx=5, pady=10)

        self.generated_latex_field = tk.Entry(self.lower_frame, width=100)
        self.generated_latex_field.grid(column=1, row=3, padx=5, pady=5)

        
        self.latex_ok = tk.BooleanVar()
        self.latex_ok.set(False)
        # Create the checkbox
        self.checkbox_latex_ok = tk.Checkbutton(self.lower_frame, text="LaTeX OK", variable=self.latex_ok, command=self.copy_latex)
        self.checkbox_latex_ok.grid(column=2, row=3, padx=5, pady=5) 
        
        # create label and textfield for "LaTeX Code Correct"
        self.correct_latex_label = tk.Label(self.lower_frame, text="LaTeX Corrected:")
        self.correct_latex_label.grid(column=0, row=4, padx=5, pady=10)
        
        self.correct_latex_field = tk.Entry(self.lower_frame,width=100, foreground="green")
        self.correct_latex_field.grid(column=1, row=4, padx=5, pady=5)
        
        # create "Generate Transcription " button
        #self.generate_transcriptions_button = tk.Button(self.lower_frame, text="Generate Transcriptions", font=("Arial",14,"bold"), command=self.generate_transcriptions,foreground="magenta")
        #self.generate_transcriptions_button.grid(column=2, row=4, padx=5, pady=5)

        # create "Load Image" button
        self.latex_preview_button = tk.Button(self.lower_frame, text="LaTeX Preview",font=("Arial",14,"bold"), command=self.latex_preview,foreground="magenta")
        self.latex_preview_button.grid(column=2, row=4, padx=5, pady=5)
        
        # ---------------------------------------------------

        self.transcription_label = tk.Label(self.lower_frame, text="Transcriptions", font=('Arial', 18, "bold"),bd=1, relief="groove",padx=40, pady=7, background="lightblue")
        self.transcription_label.grid(column=0, row=6, sticky="ew", columnspan=3)

        
        self.transcription_label_gen = tk.Label(self.lower_frame, text="Variant 1 - Generated:", anchor="e")
        self.transcription_label_gen.grid(column=0, row=8, padx=5, pady=5)
        self.transcription_var1_gen_field = tk.Entry(self.lower_frame,width=100)
        self.transcription_var1_gen_field.grid(column=1, row=8, padx=5, pady=5)

        # Create a checkbox
        self.trans1_ok = tk.BooleanVar()
        self.trans1_ok.set(False)
        self.checkbox_trans1_ok = tk.Checkbutton(self.lower_frame, text="Transcription OK", variable=self.trans1_ok, command=self.copy_trans1_ok)
        self.checkbox_trans1_ok.grid(column=2, row=8, padx=5, pady=5)

        self.transcription_label_cor = tk.Label(self.lower_frame, text="Variant 1 - Corrected:")
        self.transcription_label_cor.grid(column=0, row=9, padx=5, pady=5)
        self.transcription_var1_cor_field = tk.Entry(self.lower_frame,width=100, foreground="green")
        self.transcription_var1_cor_field.grid(column=1, row=9, padx=5, pady=5)
        
        separator = ttk.Separator(self.lower_frame, orient='horizontal')
        separator.grid(row=10, column=0, sticky='ew', padx=5, pady=5, columnspan=3)

        self.transcription_label_gen = tk.Label(self.lower_frame, text="Variant 2 - Generated:")
        self.transcription_label_gen.grid(column=0, row=11, padx=5, pady=5)
        self.transcription_var2_gen_field = tk.Entry(self.lower_frame,width=100)
        self.transcription_var2_gen_field.grid(column=1, row=11, padx=5, pady=5)

        # Create a checkbox
        self.trans2_ok = tk.BooleanVar()
        self.trans2_ok.set(False)
        self.checkbox_trans2_ok = tk.Checkbutton(self.lower_frame, text="Transcription OK", variable=self.trans2_ok, command=self.copy_trans2_ok)
        self.checkbox_trans2_ok.grid(column=2, row=11, padx=5, pady=5)

        self.transcription_label_cor = tk.Label(self.lower_frame, text="Variant 2 - Corrected:")
        self.transcription_label_cor.grid(column=0, row=12, padx=5, pady=5)
        self.transcription_var2_cor_field = tk.Entry(self.lower_frame,width=100,foreground="green")
        self.transcription_var2_cor_field.grid(column=1, row=12, padx=5, pady=5)

        separator = ttk.Separator(self.lower_frame, orient='horizontal')
        separator.grid(row=13, column=0, sticky='ew', padx=5, pady=5, columnspan=3)

        self.transcription_label_gen = tk.Label(self.lower_frame, text="Variant 3 - Generated:")
        self.transcription_label_gen.grid(column=0, row=14, padx=5, pady=5)
        self.transcription_var3_gen_field = tk.Entry(self.lower_frame,width=100)
        self.transcription_var3_gen_field.grid(column=1, row=14, padx=5, pady=5)

        # Create a checkbox
        self.trans3_ok = tk.BooleanVar()
        self.trans3_ok.set(False)
        self.checkbox_trans3_ok = tk.Checkbutton(self.lower_frame, text="Transcription OK", variable=self.trans3_ok, command=self.copy_trans3_ok)
        self.checkbox_trans3_ok.grid(column=2, row=14, padx=5, pady=5)

        self.transcription_label_cor = tk.Label(self.lower_frame, text="Variant 3 - Corrected:")
        self.transcription_label_cor.grid(column=0, row=15, padx=5, pady=5)
        self.transcription_var3_cor_field = tk.Entry(self.lower_frame,width=100,foreground="green")
        self.transcription_var3_cor_field.grid(column=1, row=15, padx=5, pady=5)

        separator = ttk.Separator(self.lower_frame, orient='horizontal')
        separator.grid(row=16, column=0, sticky='ew', padx=5, pady=5, columnspan=3)
        
        self.transcription_label_gen = tk.Label(self.lower_frame, text="Variant 4 - Generated:")
        self.transcription_label_gen.grid(column=0, row=17, padx=5, pady=5)
        self.transcription_var4_gen_field = tk.Entry(self.lower_frame,width=100)
        self.transcription_var4_gen_field.grid(column=1, row=17, padx=5, pady=5)

        # Create a checkbox
        self.trans4_ok = tk.BooleanVar()
        self.trans4_ok.set(False)
        self.checkbox_trans4_ok = tk.Checkbutton(self.lower_frame, text="Transcription OK", variable=self.trans4_ok, command=self.copy_trans4_ok)
        self.checkbox_trans4_ok.grid(column=2, row=17, padx=5, pady=5)

        self.transcription_label_cor = tk.Label(self.lower_frame, text="Variant 4 - Corrected:")
        self.transcription_label_cor.grid(column=0, row=18, padx=5, pady=5)
        self.transcription_var4_cor_field = tk.Entry(self.lower_frame,width=100,foreground="green")
        self.transcription_var4_cor_field.grid(column=1, row=18, padx=5, pady=5)

        separator = ttk.Separator(self.lower_frame, orient='horizontal')
        separator.grid(row=19, column=0, sticky='ew', padx=5, pady=5, columnspan=3)
        
        self.transcription_label_gen = tk.Label(self.lower_frame, text="Variant 5 - Generated:")
        self.transcription_label_gen.grid(column=0, row=20, padx=5, pady=5)
        self.transcription_var5_gen_field = tk.Entry(self.lower_frame,width=100)
        self.transcription_var5_gen_field.grid(column=1, row=20, padx=5, pady=5)

        # Create a checkbox
        self.trans5_ok = tk.BooleanVar()
        self.trans5_ok.set(False)
        self.checkbox_trans5_ok = tk.Checkbutton(self.lower_frame, text="Transcription OK", variable=self.trans5_ok, command=self.copy_trans5_ok)
        self.checkbox_trans5_ok.grid(column=2, row=20, padx=5, pady=5)

        self.transcription_label_cor = tk.Label(self.lower_frame, text="Variant 5 - Corrected:")
        self.transcription_label_cor.grid(column=0, row=21, padx=5, pady=5)
        self.transcription_var5_cor_field = tk.Entry(self.lower_frame,width=100, foreground="green")
        self.transcription_var5_cor_field.grid(column=1, row=21, padx=5, pady=5)

        separator = ttk.Separator(self.lower_frame, orient='horizontal')
        separator.grid(row=22, column=0, sticky='ew', padx=5, pady=5, columnspan=3)
    # ---------------------------------------------------

        # create "Load Image" button
        self.load_button = tk.Button(self.lower_frame, text="Load Data from File", font=("Arial",14,"bold"), command=self.load_data)
        self.load_button.grid(column=0, row=23, padx=5, pady=5)

        # create "Next Entry" button
        self.load_button = tk.Button(self.lower_frame, text="Next Entry", font=("Arial",14,"bold"), command=self.next_entry)
        self.load_button.grid(column=1, row=23, padx=5, pady=5)

        
        # create "Save Annotation" button
        self.save_corr_button = tk.Button(self.lower_frame, text="Save all Corrections to File", font=("Arial",14,"bold"),command=self.save_corrections, foreground="green")
        self.save_corr_button.grid(column=2, row=23, padx=5, pady=5)

    # ---------------------------------------------------

    def load_data(self):
        file_path_name = filedialog.askopenfilename()
        
        with open(file_path_name) as f:
            self.data = json.load(f)

        self.update_text_fields(0)

    # ---------------------------------------------------
    def update_text_fields(self, index):

        IMG_DIR = "images_formulas_test/"

        file_path = IMG_DIR+self.data[index]["image_name"]

        if file_path:
            img = Image.open(file_path)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

        self.filename_field.delete(0, tk.END)
        self.filename_field.insert(0,self.data[index]["image_name"])

        self.generated_latex_field.configure(state='normal')
        self.generated_latex_field.delete(0,tk.END)
        self.generated_latex_field.insert(0,self.data[index]["formula"])
        self.generated_latex_field.configure(state='readonly', readonlybackground='white')

        self.correct_latex_field.delete(0,tk.END)

        self.transcription_var1_gen_field.configure(state='normal')
        self.transcription_var1_cor_field.delete(0,tk.END)
        self.transcription_var1_gen_field.delete(0,tk.END)
        self.transcription_var1_gen_field.insert(0,self.data[index]["transcription1"])
        self.transcription_var1_gen_field.configure(state='readonly', readonlybackground='white')

        self.transcription_var2_cor_field.configure(state='normal')
        self.transcription_var2_cor_field.delete(0,tk.END)
        self.transcription_var2_gen_field.delete(0,tk.END)
        self.transcription_var2_gen_field.insert(0,self.data[index]["transcription2"])
        self.transcription_var2_gen_field.configure(state='readonly', readonlybackground='white')

        self.transcription_var3_cor_field.configure(state='normal')
        self.transcription_var3_cor_field.delete(0,tk.END)
        self.transcription_var3_gen_field.delete(0,tk.END)
        self.transcription_var3_gen_field.insert(0,self.data[index]["transcription3"])
        self.transcription_var3_gen_field.configure(state='readonly', readonlybackground='white')

        self.transcription_var4_cor_field.configure(state='normal')
        self.transcription_var4_cor_field.delete(0,tk.END)
        self.transcription_var4_gen_field.delete(0,tk.END)
        self.transcription_var4_gen_field.insert(0,self.data[index]["transcription4"])
        self.transcription_var4_gen_field.configure(state='readonly', readonlybackground='white')

        self.transcription_var4_cor_field.configure(state='normal')
        self.transcription_var5_cor_field.delete(0,tk.END)
        self.transcription_var5_gen_field.delete(0,tk.END)
        self.transcription_var5_gen_field.insert(0,self.data[index]["transcription5"])
        self.transcription_var5_gen_field.configure(state='readonly', readonlybackground='white')

        self.latex_ok.set(False)
        self.trans1_ok.set(False)
        self.trans2_ok.set(False)
        self.trans3_ok.set(False)
        self.trans4_ok.set(False)
        self.trans5_ok.set(False)
        self.image_label_preview.config(image="")

    # Function to display a dialog with the provided message
    def show_warning_dialog(self, message):
        tk.messagebox.showwarning("Warning", message)

    # ---------------------------------------------------
    def next_entry(self):
        
        cur_data = self.get_data_from_dialog()
            
        self.generated_latex_field.configure(state='normal') 
        self.transcription_var1_gen_field.configure(state='normal')
        self.transcription_var2_gen_field.configure(state='normal')
        self.transcription_var3_gen_field.configure(state='normal')
        self.transcription_var4_gen_field.configure(state='normal')
        self.transcription_var5_gen_field.configure(state='normal')   
            
        # Get the current index in the data list
        current_index = self.data.index({
            'image_name'    : self.filename_field.get(),
            'formula'       : self.generated_latex_field.get(),
            'transcription1': self.transcription_var1_gen_field.get(),
            'transcription2': self.transcription_var2_gen_field.get(),
            'transcription3': self.transcription_var3_gen_field.get(),
            'transcription4': self.transcription_var4_gen_field.get(),
            'transcription5': self.transcription_var5_gen_field.get()
        })
        print("-->", current_index)
        # Update the text fields with the values from the next index in the data list
        next_index = (current_index + 1) % len(self.data)
        self.update_text_fields(next_index)

            #self.save_corr_button.configure(foreground='green')


    # ---------------------------------------------------

    def find_entry_by_filename(self, filename):
        for entry in self.data:
            if entry['image_name'] == filename:
                return entry
        return None

    # ---------------------------------------------------
    def next_entry_save(self):

        #self.corrected_data = []

        cur_data = self.get_data_from_dialog()
        #print("cur_data: ", cur_data)

        empty_values = [key for key, value in cur_data.items() if value == '' or value is False]
        if empty_values:
            gui.show_warning_dialog("No data available!")
        else:
            cur_data = self.get_data_from_dialog()

            # Check if data is not empty
            if cur_data:
            # Add the data to the correct_data list
                self.corrected_data.append(cur_data)
            else:
                # Show info dialog if no data was selected
                self.show_info_dialog("No data was selected.")

            self.generated_latex_field.configure(state='normal') 
            self.transcription_var1_gen_field.configure(state='normal')
            self.transcription_var2_gen_field.configure(state='normal')
            self.transcription_var3_gen_field.configure(state='normal')
            self.transcription_var4_gen_field.configure(state='normal')
            self.transcription_var5_gen_field.configure(state='normal')  

            # Get the current index in the data list
            current_index = self.data.index({
                'image_name'    : self.filename_field.get(),
                'formula'       : self.generated_latex_field.get(),
                'transcription1': self.transcription_var1_gen_field.get(),
                'transcription2': self.transcription_var2_gen_field.get(),
                'transcription3': self.transcription_var3_gen_field.get(),
                'transcription4': self.transcription_var4_gen_field.get(),
                'transcription5': self.transcription_var5_gen_field.get()
            })
            print("-->", current_index)
            # Update the text fields with the values from the next index in the data list
            next_index = (current_index + 1) % len(self.data)
            self.update_text_fields(next_index)

            self.save_corr_button.configure(foreground='green')

    # ---------------------------------------------------


    def load_image(self):
        # open file dialog to select image
        file_path = filedialog.askopenfilename()
        file_name = file_path.split("/")[-1]
    
        # load image and display it in the image label
        if file_path:
            img = Image.open(file_path)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk

            # Display the file name in the filename textfield
            self.filename_field.delete(0, tk.END)
            self.filename_field.insert(0,file_path)
            #self.filename_field = file_label.pack(side="top")

    # ---------------------------------------------------


    def latex_preview(self):
        latex_code = self.correct_latex_field.get()
        # create a matplotlib figure and render the formula
        fig = Figure(figsize=(5, 1), dpi=100)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "${}$".format(latex_code), fontsize=22, ha='center', va='center', color="blue")
        ax.axis('off')

        # convert the figure to a Tkinter-compatible format
        canvas = FigureCanvasTkAgg(fig)
        canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = tk.PhotoImage(data=buf.getvalue())

        # update the label with the rendered formula
        self.image_label_preview.config(image=img)
        self.image_label_preview.image = img


    def copy_latex(self):
        if self.latex_ok.get() == 1:
            self.correct_latex_field.delete(0, tk.END) # Clear the destination text field
            self.correct_latex_field.insert(0, self.generated_latex_field.get()) # Copy the contents

    
    def copy_trans1_ok(self):
        if self.trans1_ok.get() == 1:
            self.transcription_var1_cor_field.delete(0, tk.END) # Clear the destination text field
            self.transcription_var1_cor_field.insert(0, self.transcription_var1_gen_field.get()) # Copy the contents


    def copy_trans2_ok(self):
        if self.trans2_ok.get() == 1:
            self.transcription_var2_cor_field.delete(0, tk.END) # Clear the destination text field
            self.transcription_var2_cor_field.insert(0, self.transcription_var2_gen_field.get()) # Copy the contents

    
    def copy_trans3_ok(self):
        if self.trans3_ok.get() == 1:
            self.transcription_var3_cor_field.delete(0, tk.END) # Clear the destination text field
            self.transcription_var3_cor_field.insert(0, self.transcription_var3_gen_field.get()) # Copy the contents

    def copy_trans4_ok(self):
        if self.trans4_ok.get() == 1:
            self.transcription_var4_cor_field.delete(0, tk.END) # Clear the destination text field
            self.transcription_var4_cor_field.insert(0, self.transcription_var4_gen_field.get()) # Copy the contents

    def copy_trans5_ok(self):
        if self.trans5_ok.get() == 1:
            self.transcription_var5_cor_field.delete(0, tk.END) # Clear the destination text field
            self.transcription_var5_cor_field.insert(0, self.transcription_var5_gen_field.get()) # Copy the contents

    # ---------------------------------------------------

    def save_corrections(self):

        # Ask user to choose a .json filename
        save_file_path_name = filedialog.asksaveasfilename(filetypes=[("JSON Files", "*.json")])
        print(save_file_path_name)

        # Check if a file was selected
        if save_file_path_name:
            # Check if the chosen file already exists
            if os.path.isfile(save_file_path_name):
                # File already exists, ask user for confirmation
                confirm_override = tk.messagebox.askyesno("File Already Exists", "The file already exists. Do you want to override it?")
                if not confirm_override:
                    # User chose not to override, ask for a new file name
                    save_file_path_name = filedialog.asksaveasfilename(filetypes=[("JSON Files", "*.json")])

        
        # Proceed only if a file name is selected
        if save_file_path_name:
            # Create the JSON file
            with open(save_file_path_name, 'w') as file:
                # Write JSON data
                json.dump(self.corrected_data, file)
            
            print("JSON file saved successfully.")
            self.save_corr_button.configure(foreground='red')

    # ---------------------------------------------------

    def get_data_from_dialog(self) -> list:

        # Collect data from the GUI
        image_path = self.filename_field.get()
        equation_text_cor = self.correct_latex_field.get()
        equation_text_gen = self.generated_latex_field.get()

        trans1_gen = self.transcription_var1_gen_field.get()
        trans1_cor = self.transcription_var1_cor_field.get()

        trans2_gen = self.transcription_var2_gen_field.get()
        trans2_cor = self.transcription_var2_cor_field.get()

        trans3_gen = self.transcription_var3_gen_field.get()
        trans3_cor = self.transcription_var3_cor_field.get()

        trans4_gen = self.transcription_var4_gen_field.get()
        trans4_cor = self.transcription_var4_cor_field.get()

        trans5_gen = self.transcription_var5_gen_field.get()
        trans5_cor = self.transcription_var5_cor_field.get()

        
        latex_ok = self.latex_ok.get()
        t1_ok = self.trans1_ok.get()
        t2_ok = self.trans2_ok.get()
        t3_ok = self.trans3_ok.get()
        t4_ok = self.trans4_ok.get()
        t5_ok = self.trans5_ok.get()

        cur_data={
            "image_path": image_path,
            "formula":equation_text_cor,
            "transcription1":trans1_cor,
            "transcription2":trans2_cor,
            "transcription3":trans3_cor,
            "transcription4":trans4_cor,
            "transcription5":trans5_cor,
            "formula_check":latex_ok,
            "formula_gen":equation_text_gen,
            "transcription1_gen":trans1_gen,
            "transcription2_gen":trans2_gen,
            "transcription3_gen":trans3_gen,
            "transcription4_gen":trans4_gen,
            "transcription5_gen":trans5_gen,
            "transcription_check":[t1_ok,t2_ok,t3_ok,t4_ok,t5_ok]
        }
        return cur_data

    # ---------------------------------------------------

root = tk.Tk()
gui = GUI(root)
root.mainloop()
