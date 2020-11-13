using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.QuickDataCheck
{
	/// <summary>
	/// Script form.
	/// </summary>
	/// <remarks>
	/// <para>Data management and analysis scripts can be edited and run in this form.</para>
	/// <para>Any script action is handled as if it were executed interactively by the user on the <see cref="SySal.Executables.QuickDataCheck.ExeForm">ExeForm</see>.</para>
	/// <para>C-like syntax and Pascal-like syntax are supported.</para>
	/// <para>Specific help for the various functions available is displayed by clicking on each possible function name on the right panel.</para>
	/// <para>Scripts can be saved and loaded.</para>
	/// </remarks>
	public class ScriptEditor : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button LoadButton;
		private System.Windows.Forms.TextBox ScriptText;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.TextBox OutputText;
		private System.Windows.Forms.Button ExecButton;
		private System.Windows.Forms.Button SaveButton;
		private System.Windows.Forms.Button HideButton;
		private System.Windows.Forms.GroupBox groupBox2;
		private System.Windows.Forms.ListBox FunctionList;
		private System.Windows.Forms.ToolTip SyntaxToolTip;
		private System.Windows.Forms.RadioButton SyntaxCButton;
		private System.Windows.Forms.RadioButton SyntaxPascalButton;
		private System.ComponentModel.IContainer components;

		public ScriptEditor()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.components = new System.ComponentModel.Container();
			this.LoadButton = new System.Windows.Forms.Button();
			this.ScriptText = new System.Windows.Forms.TextBox();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.OutputText = new System.Windows.Forms.TextBox();
			this.ExecButton = new System.Windows.Forms.Button();
			this.SaveButton = new System.Windows.Forms.Button();
			this.HideButton = new System.Windows.Forms.Button();
			this.groupBox2 = new System.Windows.Forms.GroupBox();
			this.FunctionList = new System.Windows.Forms.ListBox();
			this.SyntaxToolTip = new System.Windows.Forms.ToolTip(this.components);
			this.SyntaxCButton = new System.Windows.Forms.RadioButton();
			this.SyntaxPascalButton = new System.Windows.Forms.RadioButton();
			this.groupBox1.SuspendLayout();
			this.groupBox2.SuspendLayout();
			this.SuspendLayout();
			// 
			// LoadButton
			// 
			this.LoadButton.Location = new System.Drawing.Point(8, 8);
			this.LoadButton.Name = "LoadButton";
			this.LoadButton.Size = new System.Drawing.Size(48, 24);
			this.LoadButton.TabIndex = 0;
			this.LoadButton.Text = "&Load";
			this.LoadButton.Click += new System.EventHandler(this.LoadButton_Click);
			// 
			// ScriptText
			// 
			this.ScriptText.Location = new System.Drawing.Point(72, 8);
			this.ScriptText.Multiline = true;
			this.ScriptText.Name = "ScriptText";
			this.ScriptText.Size = new System.Drawing.Size(408, 184);
			this.ScriptText.TabIndex = 2;
			this.ScriptText.Text = "";
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.OutputText});
			this.groupBox1.Location = new System.Drawing.Point(8, 200);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(480, 136);
			this.groupBox1.TabIndex = 2;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Output";
			// 
			// OutputText
			// 
			this.OutputText.Location = new System.Drawing.Point(8, 16);
			this.OutputText.Multiline = true;
			this.OutputText.Name = "OutputText";
			this.OutputText.ScrollBars = System.Windows.Forms.ScrollBars.Both;
			this.OutputText.Size = new System.Drawing.Size(464, 112);
			this.OutputText.TabIndex = 2;
			this.OutputText.Text = "";
			// 
			// ExecButton
			// 
			this.ExecButton.Location = new System.Drawing.Point(8, 168);
			this.ExecButton.Name = "ExecButton";
			this.ExecButton.Size = new System.Drawing.Size(48, 24);
			this.ExecButton.TabIndex = 3;
			this.ExecButton.Text = "E&xec";
			this.ExecButton.Click += new System.EventHandler(this.ExecButton_Click);
			// 
			// SaveButton
			// 
			this.SaveButton.Location = new System.Drawing.Point(8, 40);
			this.SaveButton.Name = "SaveButton";
			this.SaveButton.Size = new System.Drawing.Size(48, 24);
			this.SaveButton.TabIndex = 1;
			this.SaveButton.Text = "&Save";
			this.SaveButton.Click += new System.EventHandler(this.SaveButton_Click);
			// 
			// HideButton
			// 
			this.HideButton.Location = new System.Drawing.Point(8, 136);
			this.HideButton.Name = "HideButton";
			this.HideButton.Size = new System.Drawing.Size(48, 24);
			this.HideButton.TabIndex = 4;
			this.HideButton.Text = "H&ide";
			this.HideButton.Click += new System.EventHandler(this.HideButton_Click);
			// 
			// groupBox2
			// 
			this.groupBox2.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.FunctionList});
			this.groupBox2.Location = new System.Drawing.Point(496, 0);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.Size = new System.Drawing.Size(152, 336);
			this.groupBox2.TabIndex = 5;
			this.groupBox2.TabStop = false;
			this.groupBox2.Text = "Functions";
			// 
			// FunctionList
			// 
			this.FunctionList.Location = new System.Drawing.Point(8, 16);
			this.FunctionList.Name = "FunctionList";
			this.FunctionList.Size = new System.Drawing.Size(136, 316);
			this.FunctionList.Sorted = true;
			this.FunctionList.TabIndex = 0;
			this.FunctionList.SelectedIndexChanged += new System.EventHandler(this.OnFunctionSelChanged);
			// 
			// SyntaxCButton
			// 
			this.SyntaxCButton.Location = new System.Drawing.Point(8, 72);
			this.SyntaxCButton.Name = "SyntaxCButton";
			this.SyntaxCButton.Size = new System.Drawing.Size(48, 24);
			this.SyntaxCButton.TabIndex = 6;
			this.SyntaxCButton.Text = "C";
			// 
			// SyntaxPascalButton
			// 
			this.SyntaxPascalButton.Location = new System.Drawing.Point(8, 96);
			this.SyntaxPascalButton.Name = "SyntaxPascalButton";
			this.SyntaxPascalButton.Size = new System.Drawing.Size(64, 24);
			this.SyntaxPascalButton.TabIndex = 7;
			this.SyntaxPascalButton.Text = "Pascal";
			// 
			// ScriptEditor
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(658, 336);
			this.ControlBox = false;
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.SyntaxPascalButton,
																		  this.SyntaxCButton,
																		  this.groupBox2,
																		  this.HideButton,
																		  this.SaveButton,
																		  this.ExecButton,
																		  this.ScriptText,
																		  this.LoadButton,
																		  this.groupBox1});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "ScriptEditor";
			this.Text = "Script Editor";
			this.Load += new System.EventHandler(this.ScriptEditor_Load);
			this.groupBox1.ResumeLayout(false);
			this.groupBox2.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		private void LoadButton_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog dlg = new OpenFileDialog();
			dlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
			dlg.CheckFileExists = true;
			dlg.Title = "Select script file";
			if (dlg.ShowDialog() == DialogResult.OK)
			{
				System.IO.StreamReader rscr = null;
				try
				{
					rscr = new System.IO.StreamReader(dlg.FileName);
					ScriptText.Text = rscr.ReadToEnd();
				}
				catch (Exception x)
				{
					MessageBox.Show(x.Message, "Script loading error!", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				if (rscr != null) rscr.Close();
			}
		}

		private void ExecButton_Click(object sender, System.EventArgs e)
		{
			try
			{
				OutputText.Clear();
				new NumericalTools.Scripting.Script(ScriptText.Text, SyntaxCButton.Checked ? NumericalTools.Scripting.Syntax.C : NumericalTools.Scripting.Syntax.Pascal).Execute();
			}
			catch (Exception x)
			{
				AddOutput(x.ToString());
			}
		}

		public void AddOutput(string s)
		{
			OutputText.Text += "\r\n" + s;
			OutputText.Select(OutputText.Text.Length, OutputText.Text.Length);
			OutputText.ScrollToCaret();			
		}

		private void SaveButton_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog dlg = new SaveFileDialog();
			dlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
			dlg.CheckPathExists = true;
			dlg.Title = "Select script file";
			if (dlg.ShowDialog() == DialogResult.OK)
			{
				System.IO.StreamWriter wscr = null;
				try
				{
					wscr = new System.IO.StreamWriter(dlg.FileName);
					wscr.Write(ScriptText.Text);
					wscr.Flush();
				}
				catch (Exception x)
				{
					MessageBox.Show(x.Message, "Script writing error!", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				if (wscr != null) wscr.Close();
			}		
		}

		private void HideButton_Click(object sender, System.EventArgs e)
		{
			Hide();
		}

		private void OnFunctionSelChanged(object sender, System.EventArgs e)
		{
			if (FunctionList.SelectedItem != null)
			{
				NumericalTools.Scripting.FunctionDescriptor d = (NumericalTools.Scripting.FunctionDescriptor)FunctionList.SelectedItem;
				string fcall = d.Name + "(";
				int i;
				for (i = 0; i < d.Parameters.Length; i++)
				{
					NumericalTools.Scripting.ParameterDescriptor pd = d.Parameters[i];
					if (i > 0) fcall += ", ";
					fcall += pd.Type.ToString() + " " + pd.Name;
				}
				fcall += ")\r\n";
				SyntaxToolTip.SetToolTip(FunctionList, fcall + "\r\n" + d.Help);
				SyntaxToolTip.Active = true;				
			}
		}

		private void ScriptEditor_Load(object sender, System.EventArgs e)
		{
			NumericalTools.Scripting.FunctionDescriptor [] fdlist = NumericalTools.Scripting.Script.GetFunctionDescriptors();
			FunctionList.Items.Clear();
			SyntaxCButton.Checked = true;
			SyntaxPascalButton.Checked = false;
			foreach (NumericalTools.Scripting.FunctionDescriptor fd in fdlist)
				FunctionList.Items.Add(fd);
		}
	}
}
