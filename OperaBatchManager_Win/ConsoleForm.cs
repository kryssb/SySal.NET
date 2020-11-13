using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Services.OperaBatchManager_Win
{
	/// <summary>
	/// ConsoleForm shows log information about an associated process operation.
	/// </summary>
	/// <remarks>
	/// <para>Every implementation of BatchManager supports a log for process operations. OperaBatchManager_Win uses logger windows, a.k.a. ConsoleForm instances, to show a text format log of each process operation activity.</para>
	/// <para>Normally, the ConsoleForm would be hidden when a process is paused, and closed when the process is completed or aborted.</para>
	/// <para>Its data are available only as long as the OperaBatchManager itself is running, being lost on OperaBatchManager shutdown.</para>
	/// <para>It is possible to save the results of the log: by specifying a proper file path to store the output, and checking the "Save on close" checkbox, the user directs the ConsoleForm to dump its content automatically to a file when it is closed.</para>
	/// </remarks>
	public class ConsoleForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.RichTextBox RTFText;
		private System.Windows.Forms.SaveFileDialog SaveTextFileDialog;
		private System.Windows.Forms.TextBox OutFileText;
		private System.Windows.Forms.CheckBox SaveOnCloseCheck;
		private System.Windows.Forms.Button SelFileButton;
		private System.Windows.Forms.Button HideButton;		
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public ConsoleForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		}

        private delegate void dSetVisible(bool vis);

        internal void SetVisible(bool vis)
        {
            if (this.InvokeRequired) this.Invoke(new dSetVisible(this.SetVisible), vis);
            else Visible = vis;
        }

        private delegate void dWrite(string s);

		internal void Write(string s)
		{
            if (this.InvokeRequired) this.Invoke(new dWrite(this.Write), s);
            else AddText(s);
		}

		void AddText(string s)
		{
			try
			{
				RTFText.AppendText(s);
			}
			catch (Exception)
			{
				System.Threading.Thread.Sleep(1000);
				try
				{
					RTFText.AppendText(s);
				}
				catch (Exception) {}
			}
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
			this.RTFText = new System.Windows.Forms.RichTextBox();
			this.SaveTextFileDialog = new System.Windows.Forms.SaveFileDialog();
			this.OutFileText = new System.Windows.Forms.TextBox();
			this.SaveOnCloseCheck = new System.Windows.Forms.CheckBox();
			this.SelFileButton = new System.Windows.Forms.Button();
			this.HideButton = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// RTFText
			// 
			this.RTFText.CausesValidation = false;
			this.RTFText.DetectUrls = false;
			this.RTFText.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.RTFText.Location = new System.Drawing.Point(8, 8);
			this.RTFText.Name = "RTFText";
			this.RTFText.ReadOnly = true;
			this.RTFText.Size = new System.Drawing.Size(704, 344);
			this.RTFText.TabIndex = 0;
			this.RTFText.Text = "";
			// 
			// SaveTextFileDialog
			// 
			this.SaveTextFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
			this.SaveTextFileDialog.Title = "Select file to save the content of the console.";
			// 
			// OutFileText
			// 
			this.OutFileText.Location = new System.Drawing.Point(144, 360);
			this.OutFileText.Name = "OutFileText";
			this.OutFileText.Size = new System.Drawing.Size(512, 20);
			this.OutFileText.TabIndex = 2;
			this.OutFileText.Text = "";
			// 
			// SaveOnCloseCheck
			// 
			this.SaveOnCloseCheck.Location = new System.Drawing.Point(8, 360);
			this.SaveOnCloseCheck.Name = "SaveOnCloseCheck";
			this.SaveOnCloseCheck.TabIndex = 3;
			this.SaveOnCloseCheck.Text = "Save on close";
			// 
			// SelFileButton
			// 
			this.SelFileButton.Location = new System.Drawing.Point(112, 360);
			this.SelFileButton.Name = "SelFileButton";
			this.SelFileButton.Size = new System.Drawing.Size(24, 24);
			this.SelFileButton.TabIndex = 4;
			this.SelFileButton.Text = "...";
			this.SelFileButton.Click += new System.EventHandler(this.SelFileButton_Click);
			// 
			// HideButton
			// 
			this.HideButton.Location = new System.Drawing.Point(664, 360);
			this.HideButton.Name = "HideButton";
			this.HideButton.Size = new System.Drawing.Size(48, 24);
			this.HideButton.TabIndex = 5;
			this.HideButton.Text = "Hide";
			this.HideButton.Click += new System.EventHandler(this.HideButton_Click);
			// 
			// ConsoleForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(722, 392);
			this.ControlBox = false;
			this.Controls.Add(this.HideButton);
			this.Controls.Add(this.SelFileButton);
			this.Controls.Add(this.SaveOnCloseCheck);
			this.Controls.Add(this.OutFileText);
			this.Controls.Add(this.RTFText);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.Name = "ConsoleForm";
			this.Text = "Console";
			this.Closing += new System.ComponentModel.CancelEventHandler(this.OnClosing);
			this.ResumeLayout(false);

		}
		#endregion

		private void SelFileButton_Click(object sender, System.EventArgs e)
		{
			if (SaveTextFileDialog.ShowDialog() == DialogResult.OK)
				OutFileText.Text = SaveTextFileDialog.FileName;					
		}

		private void OnClosing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			if (SaveOnCloseCheck.Checked)
				try
				{
					RTFText.SaveFile(OutFileText.Text, System.Windows.Forms.RichTextBoxStreamType.PlainText);
				}
				catch (Exception) {}
		}

		internal void ShowDlg() 		
		{
            MainForm.ThreadLogStart("ShowDlg");
            //ShowDialog(OperaBatchManager_Win.MainForm.TheMainForm); 
            //Visible = true;
            Application.Run(this);
            MainForm.ThreadLogEnd();
        }

		private void HideButton_Click(object sender, System.EventArgs e)
		{
            Visible = false;
		}
	}
}
