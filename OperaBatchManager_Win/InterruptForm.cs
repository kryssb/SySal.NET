using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Services.OperaBatchManager_Win
{
	/// <summary>
	/// InterruptForm is used to specify a data-string for a new interrupt.
	/// </summary>
	/// <remarks>
	/// <para>When the user is ready to send an interrupt to an operation, he/she selects the operation in the MainForm, and then clicks the Interrupt button. On doing so, the InterruptForm is opened, and interrupt data can be entered.</para>
	/// <para>The format of the interrupt data string is completely free, as it depends exclusively on the driver executable that is to receive these data.</para>
	/// <para>A null interrupt (interrupt with a null data-string) is of little use, although it may not be excluded that some driver uses such signals (e.g. for synchronization or similar tasks).</para>
	/// <para>Refer to the documentation of the specific driver being interrupted for its supported interrupts and their syntax.</para>
	/// </remarks>
	public class InterruptForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button SendButton;
		private System.Windows.Forms.Button CancelButton;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		internal System.Windows.Forms.RichTextBox RTFIn;
		private System.Windows.Forms.GroupBox InterruptDataGroup;

		private Size InterruptDataDeflate;
		private Size GroupBoxDeflate;

		public InterruptForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			InterruptDataDeflate.Width = this.Width - RTFIn.Width;
			InterruptDataDeflate.Height = this.Height - RTFIn.Height;
			GroupBoxDeflate.Width = this.Width - InterruptDataGroup.Width;
			GroupBoxDeflate.Height = this.Height - InterruptDataGroup.Height;
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
			this.SendButton = new System.Windows.Forms.Button();
			this.CancelButton = new System.Windows.Forms.Button();
			this.RTFIn = new System.Windows.Forms.RichTextBox();
			this.InterruptDataGroup = new System.Windows.Forms.GroupBox();
			this.InterruptDataGroup.SuspendLayout();
			this.SuspendLayout();
			// 
			// SendButton
			// 
			this.SendButton.Location = new System.Drawing.Point(528, 8);
			this.SendButton.Name = "SendButton";
			this.SendButton.Size = new System.Drawing.Size(56, 24);
			this.SendButton.TabIndex = 0;
			this.SendButton.Text = "Send";
			this.SendButton.Click += new System.EventHandler(this.SendButton_Click);
			// 
			// CancelButton
			// 
			this.CancelButton.Location = new System.Drawing.Point(8, 8);
			this.CancelButton.Name = "CancelButton";
			this.CancelButton.Size = new System.Drawing.Size(56, 24);
			this.CancelButton.TabIndex = 1;
			this.CancelButton.Text = "Cancel";
			this.CancelButton.Click += new System.EventHandler(this.CancelButton_Click);
			// 
			// RTFIn
			// 
			this.RTFIn.Font = new System.Drawing.Font("Lucida Console", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.RTFIn.Location = new System.Drawing.Point(8, 16);
			this.RTFIn.Name = "RTFIn";
			this.RTFIn.Size = new System.Drawing.Size(560, 200);
			this.RTFIn.TabIndex = 2;
			this.RTFIn.Text = "";
			// 
			// InterruptDataGroup
			// 
			this.InterruptDataGroup.Controls.Add(this.RTFIn);
			this.InterruptDataGroup.Location = new System.Drawing.Point(8, 40);
			this.InterruptDataGroup.Name = "InterruptDataGroup";
			this.InterruptDataGroup.Size = new System.Drawing.Size(576, 224);
			this.InterruptDataGroup.TabIndex = 3;
			this.InterruptDataGroup.TabStop = false;
			this.InterruptDataGroup.Text = "Interrupt data";
			// 
			// InterruptForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(592, 266);
			this.Controls.Add(this.CancelButton);
			this.Controls.Add(this.SendButton);
			this.Controls.Add(this.InterruptDataGroup);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
			this.MinimumSize = new System.Drawing.Size(400, 100);
			this.Name = "InterruptForm";
			this.Text = "Insert interrupt data";
			this.Resize += new System.EventHandler(this.OnResize);
			this.InterruptDataGroup.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		private void OnResize(object sender, System.EventArgs e)
		{
			RTFIn.Width = this.Width - InterruptDataDeflate.Width;
			RTFIn.Height = this.Height - InterruptDataDeflate.Height;	
			InterruptDataGroup.Width = this.Width - GroupBoxDeflate.Width;
			InterruptDataGroup.Height = this.Height - GroupBoxDeflate.Height;	
			SendButton.Location = new Point(this.Width - GroupBoxDeflate.Width - SendButton.Width, SendButton.Top);
		}

		private void SendButton_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.OK;
			Close();
		}

		private void CancelButton_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();		
		}
	}
}
