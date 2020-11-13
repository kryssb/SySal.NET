using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables
{
    public partial class StageHostProcessForm : Form
    {
        public StageHostProcessForm()
        {
            InitializeComponent();
        }

        SySal.StageControl.StageHostInterface m_HI = null;

        private void OnLoad(object sender, EventArgs e)
        {
            m_HI = new SySal.StageControl.StageHostInterface();
            System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new System.Runtime.Remoting.Channels.Tcp.TcpChannel(1881), false);
            System.Runtime.Remoting.RemotingServices.Marshal(m_HI, SySal.StageControl.StageHostInterface.ConnectString);
        }

        private void OnClosing(object sender, FormClosingEventArgs e)
        {
            System.Runtime.Remoting.RemotingServices.Disconnect(m_HI);
            m_HI.Dispose();
        }        
    }
}