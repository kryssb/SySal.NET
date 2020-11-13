using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class MarkAcquisitionForm : SySal.SySalNExTControls.SySalDialog
    {
        MarkAcquisitionSettings C = MarkAcquisitionSettings.Default;

        internal SySal.StageControl.IStage iStage;
        internal ISySalCameraDisplay iCamDisp;
        internal IScannerDataDisplay iScanDataDisplay;

        string m_MarkString = "";

        string m_MapType = "";

        SySal.BasicTypes.Rectangle m_Extents;

        SySal.BasicTypes.Identifier m_Id;

        SySal.DAQSystem.Scanning.MountPlateDesc m_PlateDesc;

        public SySal.DAQSystem.Scanning.MountPlateDesc PlateDesc
        {
            get { return m_PlateDesc; }
            set { m_PlateDesc = value; }
        }

        class Mark
        {
            public int Id;
            public SySal.BasicTypes.Vector2 MapPos;
            public SySal.BasicTypes.Vector2 ExpectedPos;
            public SySal.BasicTypes.Vector2 FoundPos;
            public bool XValid;
            public bool YValid;
            public bool NotFound;
            public int Side;
        }

        Mark[] m_Marks = new Mark[0];

        static System.Text.RegularExpressions.Regex rx_markheadersyntax = new System.Text.RegularExpressions.Regex(@"\s*(mapX|mapext|map|mapx)\s*:\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*;\s*(\d+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)");

        static System.Text.RegularExpressions.Regex rx_marksyntax = new System.Text.RegularExpressions.Regex(@"\s*;\s*(\d+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+(\d+)\s+(\d+)\s+(\d+)");

        public MarkAcquisitionForm()
        {
            InitializeComponent();
            if (m_MarkString.Length > 0)
            {
                txtMarkString.Text = m_MarkString;
                txtMarkString.ReadOnly = true;
            }
            else txtMarkString.ReadOnly = false;
            m_Map.MXX = m_Map.MYY = 1.0;
            m_Map.MXY = m_Map.MYX = 0.0;
            m_Map.RX = m_Map.RY = 0.0;
            m_Map.TX = m_Map.TY = m_Map.TZ;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        bool ReadMarkString()
        {
            m_MarkString = txtMarkString.Text;
            System.Text.RegularExpressions.Match mh = rx_markheadersyntax.Match(m_MarkString);
            if (mh.Success == false || mh.Index != 0)
                return false;
            try
            {
                m_MapType = mh.Groups[1].Value;
                m_Id.Part0 = int.Parse(mh.Groups[2].Value);
                m_Id.Part1 = int.Parse(mh.Groups[3].Value);
                m_Id.Part2 = int.Parse(mh.Groups[4].Value);
                m_Id.Part3 = int.Parse(mh.Groups[5].Value);
                m_Marks = new Mark[int.Parse(mh.Groups[6].Value)];
                m_Extents.MinX = double.Parse(mh.Groups[7].Value, System.Globalization.CultureInfo.InvariantCulture);
                m_Extents.MinY = double.Parse(mh.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture);
                m_Extents.MaxX = double.Parse(mh.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                m_Extents.MaxY = double.Parse(mh.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                int pos = mh.Length;
                int markn = 0;
                System.Text.RegularExpressions.Match mm;
                while (pos < m_MarkString.Length && markn < m_Marks.Length)
                {
                    mm = rx_marksyntax.Match(m_MarkString, pos);
                    if (mm.Success == false) throw new Exception("Invalid syntax found at " + pos);
                    m_Marks[markn] = new Mark();
                    m_Marks[markn].Id = int.Parse(mm.Groups[1].Value);
                    m_Marks[markn].MapPos.X = double.Parse(mm.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                    m_Marks[markn].MapPos.Y = double.Parse(mm.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                    m_Marks[markn].Side = int.Parse(mm.Groups[8].Value);
                    pos += mm.Length;
                    markn++;
                }
            }
            catch (Exception x)
            {
                m_MapType = "INVALID";
                return false;
            }
            return true;
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            if (ReadMarkString() == false)
            {
                MessageBox.Show("Error in mark string syntax", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            new System.Threading.Thread(new System.Threading.ThreadStart(AcquireMarks)).Start();
        }

        enum State { Invalid, Idle, Moving, Acquiring, PauseMoving, PauseAcquiring, Done, Failed }

        bool m_IsEnteringState = false;
        State m_State;

        void SetCurrentExpectedPosition(Mark m)
        {
            var selm = m_Marks.Where(ma => ma.XValid == true && ma.YValid == true);
            switch (selm.Count())
            {
                case 0:
                    try
                    {
                        double tx = iStage.GetNamedReferencePosition("ReferenceX");
                        double ty = iStage.GetNamedReferencePosition("ReferenceY");
                        m.ExpectedPos.X = tx + m.MapPos.X;
                        m.ExpectedPos.Y = ty + m.MapPos.Y;
                    }
                    catch (Exception x)
                    {
                        MessageBox.Show(x.ToString(), "Error computing position", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        m.ExpectedPos.X = m.MapPos.X;
                        m.ExpectedPos.Y = m.MapPos.Y;
                    }
                    return;

                case 1:
                    {
                        Mark sm = selm.First();
                        m.ExpectedPos.X = m.MapPos.X + (sm.FoundPos.X - sm.MapPos.X);
                        m.ExpectedPos.Y = m.MapPos.Y + (sm.FoundPos.Y - sm.MapPos.Y);
                    }
                    return;

                case 2:
                    {
                        Mark m0 = selm.ElementAt(0);
                        Mark m1 = selm.ElementAt(1);
                        double fdx = m0.FoundPos.X - m1.FoundPos.X;
                        double fdy = m0.FoundPos.Y - m1.FoundPos.Y;
                        double mdx = m0.MapPos.X - m1.MapPos.X;
                        double mdy = m0.MapPos.Y - m1.MapPos.Y;
                        double e = Math.Sqrt(fdx * fdx + fdy * fdy) / Math.Sqrt(mdx * mdx + mdy * mdy);
                        double r = Math.Atan2(fdy, fdx) - Math.Atan2(mdy, mdx);
                        double cr = Math.Cos(r);
                        double sr = Math.Sin(r);
                        double d0x = m0.FoundPos.X - m0.MapPos.X;
                        double d0y = m0.FoundPos.Y - m0.MapPos.Y;
                        double dx = m.MapPos.X - m0.MapPos.X;
                        double dy = m.MapPos.Y - m0.MapPos.Y;
                        m.ExpectedPos.X = d0x + (dx * cr - dy * sr) * e;
                        m.ExpectedPos.Y = d0y + (dx * sr + dy * cr) * e;
                    }
                    return;

                default:
                    {
                        double[] inDX = new double[selm.Count()];
                        double[] inDY = new double[selm.Count()];
                        double[] inX = new double[selm.Count()];
                        double[] inY = new double[selm.Count()];
                        int i;
                        for (i = 0; i < selm.Count(); i++)
                        {
                            Mark s = selm.ElementAt(i);
                            inX[i] = s.MapPos.X - selm.First().MapPos.X;
                            inY[i] = s.MapPos.Y - selm.First().MapPos.Y;
                            inDX[i] = s.FoundPos.X - s.MapPos.X;
                            inDY[i] = s.FoundPos.Y - s.MapPos.Y;
                        }
                        double[] outpar = new double[7];
                        if (NumericalTools.Fitting.Affine(inDX, inDY, inX, inY, ref outpar) == NumericalTools.ComputationResult.OK)
                        {
                            double dx = m.MapPos.X - selm.First().MapPos.X;
                            double dy = m.MapPos.Y - selm.First().MapPos.Y;
                            m.ExpectedPos.X = (outpar[0] + 1.0) * dx + outpar[1] * dy + outpar[4] + selm.First().MapPos.X;
                            m.ExpectedPos.Y = outpar[2] * dx + (1.0 + outpar[3]) * dy + outpar[5] + selm.First().MapPos.Y;
                        }
                        else
                        {
                            try
                            {
                                m.ExpectedPos.X = iStage.GetNamedReferencePosition("ReferenceX") + m.MapPos.X;
                                m.ExpectedPos.Y = iStage.GetNamedReferencePosition("ReferenceY") + m.MapPos.Y;
                            }
                            catch (Exception x)
                            {
                                MessageBox.Show(x.ToString(), "Error computing position", MessageBoxButtons.OK, MessageBoxIcon.Error);
                                m.ExpectedPos.X = m.MapPos.X;
                                m.ExpectedPos.Y = m.MapPos.Y;
                            }
                        }
                    }
                    break;
            }       
        }

        delegate void dSetText(System.Windows.Forms.Control ctl, string txt);

        void SetText(System.Windows.Forms.Control ctl, string txt)
        {
            if (this.InvokeRequired) this.Invoke(new dSetText(SetText), new object[] {ctl, txt});
            else ctl.Text = txt;
        }

        delegate void dEnable(System.Windows.Forms.Control ctl, bool en);

        void Enable(System.Windows.Forms.Control ctl, bool en)
        {
            if (this.InvokeRequired) this.Invoke(new dEnable(Enable), new object[] { ctl, en });
            else ctl.Enabled = en;
        }

        enum MessageType { Null, Pause, Continue, Stop, SetX, SetY, SetNotFound, SetNotSearched, Previous, Next }

        System.Collections.Generic.Queue<MessageType> m_Messages = new Queue<MessageType>();

        void EnterState(State st)
        {
            m_IsEnteringState = true;
            m_State = st;
            switch (m_State)
            {
                case State.Acquiring:
                    {
                        iCamDisp.EnableCross = true;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, true);
                        Enable(btnContinue, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnSetX, true);
                        Enable(btnSetY, true);
                        Enable(btnSetNotFound, true);
                        Enable(btnSetNotSearched, true);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnDone, false);
                        Enable(btnCancel, true);
                    }
                    break;

                case State.Done:
                    {
                        iCamDisp.EnableCross = false;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, false);
                        Enable(btnContinue, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnSetX, false);
                        Enable(btnSetY, false);
                        Enable(btnSetNotFound, false);
                        Enable(btnSetNotSearched, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnDone, true);
                        Enable(btnCancel, true);
                    }
                    break;

                case State.Failed:
                    {
                        iCamDisp.EnableCross = false;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, false);
                        Enable(btnContinue, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnSetX, false);
                        Enable(btnSetY, false);
                        Enable(btnSetNotFound, false);
                        Enable(btnSetNotSearched, false);
                        Enable(btnPrevious, true);
                        Enable(btnNext, true);
                        Enable(btnDone, false);
                        Enable(btnCancel, true);
                    }
                    break;

                case State.Idle:
                    {
                        iCamDisp.EnableCross = false;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, false);
                        Enable(btnContinue, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnSetX, false);
                        Enable(btnSetY, false);
                        Enable(btnSetNotFound, false);
                        Enable(btnSetNotSearched, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnDone, false);
                        Enable(btnCancel, false);
                    }
                    break;

                case State.Invalid:
                    {
                        iCamDisp.EnableCross = false;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, false);
                        Enable(btnContinue, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnSetX, false);
                        Enable(btnSetY, false);
                        Enable(btnSetNotFound, false);
                        Enable(btnSetNotSearched, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnDone, false);
                        Enable(btnCancel, true);
                    }
                    break;

                case State.Moving:
                    {
                        iCamDisp.EnableCross = false;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, true);
                        Enable(btnContinue, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnSetX, false);
                        Enable(btnSetY, false);
                        Enable(btnSetNotFound, false);
                        Enable(btnSetNotSearched, false);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnDone, false);
                        Enable(btnCancel, true);
                    }
                    break;

                case State.PauseAcquiring:
                    {
                        iCamDisp.EnableCross = true;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, false);
                        Enable(btnContinue, true);
                        Enable(btnPrevious, true);
                        Enable(btnNext, true);
                        Enable(btnSetX, true);
                        Enable(btnSetY, true);
                        Enable(btnSetNotFound, true);
                        Enable(btnSetNotSearched, true);
                        Enable(btnPrevious, false);
                        Enable(btnNext, false);
                        Enable(btnDone, false);
                        Enable(btnCancel, true);
                    }
                    break;

                case State.PauseMoving:
                    {
                        iCamDisp.EnableCross = true;
                        Enable(btnStart, false);
                        Enable(btnStop, false);
                        Enable(btnPause, false);
                        Enable(btnContinue, true);
                        Enable(btnPrevious, true);
                        Enable(btnNext, true);
                        Enable(btnSetX, true);
                        Enable(btnSetY, true);
                        Enable(btnSetNotFound, true);
                        Enable(btnSetNotSearched, true);
                        Enable(btnPrevious, true);
                        Enable(btnNext, true);
                        Enable(btnDone, false);
                        Enable(btnCancel, true);
                    }
                    break;
            }
        }

        bool IsEnteringState
        {
            get
            {
                bool r = m_IsEnteringState;
                m_IsEnteringState = false;
                return r;
            }
        }

        void SyncCurrentMark(Mark CurrentMark)
        {
            SetCurrentExpectedPosition(CurrentMark);
            SetText(txtID, CurrentMark.Id.ToString());
            SetText(txtMapX, CurrentMark.MapPos.X.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
            SetText(txtMapY, CurrentMark.MapPos.Y.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
            SetText(txtExpX, CurrentMark.ExpectedPos.X.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
            SetText(txtExpY, CurrentMark.ExpectedPos.Y.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
            SetText(txtFlag, CurrentMark.NotFound ? "Not Found" : ((CurrentMark.XValid && CurrentMark.YValid) ? "Found" : "Not searched"));
            iStage.Stop(SySal.StageControl.Axis.X);
            iStage.Stop(SySal.StageControl.Axis.Y);
            iStage.Stop(SySal.StageControl.Axis.Z);
        }

        private void AcquireMarks()
        {
            int i;
            Mark CurrentMark = new Mark();
            CurrentMark.Id = -1;
            EnterState(State.Idle);
            while (m_State != State.Invalid)
            {
                MessageType msg = MessageType.Null;
                lock (m_Messages)
                    if (m_Messages.Count == 0)
                    {
                        System.Threading.Thread.Sleep(200);
                        if (m_Messages.Count == 0)
                            msg = MessageType.Null;
                        else
                            msg = m_Messages.Dequeue();
                    }
                    else msg = m_Messages.Dequeue();

                switch (m_State)
                {
                    case State.Idle:
                        if (IsEnteringState)
                        {

                        }
                        switch (msg)
                        {
                            case MessageType.Stop: EnterState(State.Failed); continue; 
                        }
                        IEnumerable<Mark> nextmarks = m_Marks.Where(m => m.NotFound == false && (m.XValid == false || m.YValid == false));
                        if (nextmarks.Count() == 0)
                            EnterState((m_Marks.Where(m => m.NotFound == false).Count() >= 2) ? State.Done : State.Failed);
                        else
                        {                            
                            CurrentMark = nextmarks.First();
                            SyncCurrentMark(CurrentMark);
                            iStage.PosMove(SySal.StageControl.Axis.X, CurrentMark.ExpectedPos.X, C.XYSpeed, C.XYAcceleration, C.XYAcceleration);
                            iStage.PosMove(SySal.StageControl.Axis.Y, CurrentMark.ExpectedPos.Y, C.XYSpeed, C.XYAcceleration, C.XYAcceleration);
                            EnterState(State.Moving);
                        }
                        break;

                    case State.Moving:
                        if (IsEnteringState)
                        {

                        }
                        switch (msg)
                        {
                            case MessageType.Stop: EnterState(State.Failed); continue;
                            case MessageType.Pause:
                                {
                                    iStage.Stop(StageControl.Axis.X);
                                    iStage.Stop(StageControl.Axis.Y);
                                    iStage.Stop(StageControl.Axis.Z);
                                    EnterState(State.PauseMoving);
                                }
                                continue;
                        }
                        {
                            if (Math.Abs(iStage.GetPos(SySal.StageControl.Axis.X) - CurrentMark.ExpectedPos.X) < C.XYPosTolerance &&
                                Math.Abs(iStage.GetPos(SySal.StageControl.Axis.Y) - CurrentMark.ExpectedPos.Y) < C.XYPosTolerance)
                            {
                                EnterState(State.Acquiring);
                            }
                        }
                        break;

                    case State.Acquiring:
                        if (IsEnteringState)
                        {

                        }
                        switch (msg)
                        {
                            case MessageType.Stop: EnterState(State.Failed); continue;
                            case MessageType.Pause: EnterState(State.PauseMoving); continue;
                            case MessageType.SetX:
                                {
                                    CurrentMark.FoundPos.X = iStage.GetPos(SySal.StageControl.Axis.X);
                                    SetText(txtFoundX, CurrentMark.FoundPos.X.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
                                    CurrentMark.XValid = true;
                                    CurrentMark.NotFound = false;
                                    SetText(txtFlag, (CurrentMark.XValid && CurrentMark.YValid) ? "Found" : "Not searched");
                                    if (CurrentMark.YValid) EnterState(State.Idle);
                                }
                                break;
                            case MessageType.SetY:
                                {
                                    CurrentMark.FoundPos.Y = iStage.GetPos(SySal.StageControl.Axis.Y);
                                    SetText(txtFoundY, CurrentMark.FoundPos.Y.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
                                    CurrentMark.YValid = true;
                                    CurrentMark.NotFound = false;
                                    SetText(txtFlag, (CurrentMark.XValid && CurrentMark.YValid) ? "Found" : "Not searched");
                                    if (CurrentMark.XValid) EnterState(State.Idle);
                                }
                                break;
                            case MessageType.SetNotFound:
                                {
                                    CurrentMark.XValid = CurrentMark.YValid = false;
                                    CurrentMark.NotFound = true;
                                    SetText(txtFlag, "Not Found");
                                    EnterState(State.Idle);
                                }
                                break;
                            case MessageType.SetNotSearched:
                                {
                                    CurrentMark.XValid = CurrentMark.YValid = false;
                                    SetText(txtFlag, "Not Searched");
                                }
                                break;
                        }
                        {
                            /* no action here */
                        }
                        break;

                    case State.PauseMoving:
                        if (IsEnteringState)
                        {

                        }
                        switch (msg)
                        {
                            case MessageType.Continue:
                                {
                                    if (CurrentMark.NotFound || (CurrentMark.XValid == true && CurrentMark.YValid == true))
                                        EnterState(State.Idle);
                                    else
                                    {
                                        iStage.PosMove(SySal.StageControl.Axis.X, CurrentMark.ExpectedPos.X, C.XYSpeed, C.XYAcceleration, C.XYAcceleration);
                                        iStage.PosMove(SySal.StageControl.Axis.Y, CurrentMark.ExpectedPos.Y, C.XYSpeed, C.XYAcceleration, C.XYAcceleration);
                                        EnterState(State.Moving);
                                    }
                                }
                                continue;
                            case MessageType.Stop: EnterState(State.Failed); continue;                            
                            case MessageType.SetX:
                                {
                                    CurrentMark.FoundPos.X = iStage.GetPos(SySal.StageControl.Axis.X);
                                    SetText(txtFoundX, CurrentMark.FoundPos.X.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
                                    CurrentMark.XValid = true;
                                    CurrentMark.NotFound = false;
                                    SetText(txtFlag, (CurrentMark.XValid && CurrentMark.YValid) ? "Found" : "Not searched");
                                    if (CurrentMark.YValid) EnterState(State.Idle);
                                }
                                break;
                            case MessageType.SetY:
                                {
                                    CurrentMark.FoundPos.Y = iStage.GetPos(SySal.StageControl.Axis.Y);
                                    SetText(txtFoundY, CurrentMark.FoundPos.Y.ToString("F0", System.Globalization.CultureInfo.InvariantCulture));
                                    CurrentMark.YValid = true;
                                    CurrentMark.NotFound = false;
                                    SetText(txtFlag, (CurrentMark.XValid && CurrentMark.YValid) ? "Found" : "Not searched");
                                    if (CurrentMark.XValid) EnterState(State.Idle);
                                }
                                break;
                            case MessageType.SetNotFound:
                                {
                                    CurrentMark.XValid = CurrentMark.YValid = false;
                                    CurrentMark.NotFound = true;
                                    SetText(txtFlag, "Not Found");
                                    EnterState(State.Idle);
                                }
                                break;
                            case MessageType.SetNotSearched:
                                {
                                    CurrentMark.XValid = CurrentMark.YValid = false;
                                    SetText(txtFlag, "Not Searched");
                                }
                                break;
                            case MessageType.Next:
                                {
                                    int im;
                                    for (im = 0; m_Marks[im] != CurrentMark; im++) ;
                                    if (im < m_Marks.Length - 1)
                                    {
                                        CurrentMark = m_Marks[im + 1];
                                        SyncCurrentMark(CurrentMark);                                        
                                    }
                                }
                                break;
                            case MessageType.Previous:
                                {
                                    int im;
                                    for (im = 0; m_Marks[im] != CurrentMark; im++) ;
                                    if (im > 0)
                                    {
                                        CurrentMark = m_Marks[im - 1];
                                        SyncCurrentMark(CurrentMark);                                        
                                    }
                                }
                                break;
                        }
                        {
                            /* no action here */
                        }
                        break;

                    case State.PauseAcquiring:
                        if (IsEnteringState)
                        {

                        }
                        switch (msg)
                        {
                            case MessageType.Stop: EnterState(State.Failed); continue;                            
                        }
                        EnterState(State.Acquiring);
                        break;

                    case State.Done:
                        if (IsEnteringState)
                        {
                            iCamDisp.EnableCross = false;
                        }
                        return;                        

                    case State.Failed:
                        if (IsEnteringState)
                        {
                            iCamDisp.EnableCross = false;
                        }
                        return;                        

                    default: 
                        m_State = State.Invalid;
                        iCamDisp.EnableCross = false;
                        return;
                }
            }
        }

        private void btnSetX_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.SetX);
        }

        private void btnSetY_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.SetY);
        }

        private void btnSetNotFound_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.SetNotFound);
        }

        private void btnSetNotSearched_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.SetNotSearched);
        }

        private void btnPrevious_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.Previous);
        }

        private void btnNext_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.Next);
        }

        private void btnPause_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.Pause);
        }

        private void btnContinue_Click(object sender, EventArgs e)
        {
            lock (m_Messages)
                m_Messages.Enqueue(MessageType.Continue);
        }

        private void btnDone_Click(object sender, EventArgs e)
        {
            ComputeMap();
            DialogResult = DialogResult.OK;
            Close();
        }

        SySal.DAQSystem.Scanning.IntercalibrationInfo m_Map;

        public SySal.DAQSystem.Scanning.IntercalibrationInfo Map
        {
            get
            {
                return m_Map;
            }
        }

        void ComputeMap()
        {
            var selm = m_Marks.Where(ma => ma.XValid == true && ma.YValid == true);
            switch (selm.Count())
            {
                case 0:
                    {
                        m_Map.MXX = m_Map.MYY = 1.0;
                        m_Map.MXY = m_Map.MYX = 0.0;
                        m_Map.TX = m_Map.TY = 0.0;
                        m_Map.RX = m_Map.RY = 0.0;
                    }
                    return;

                case 1:
                    {
                        Mark sm = selm.First();
                        m_Map.MXX = m_Map.MYY = 1.0;
                        m_Map.MXY = m_Map.MYX = 0.0;
                        m_Map.TX = sm.FoundPos.X - sm.MapPos.X;
                        m_Map.TY = sm.FoundPos.Y - sm.MapPos.Y;
                        m_Map.RX = m_Map.RY = 0.0;
                    }
                    return;

                case 2:
                    {
                        Mark m0 = selm.ElementAt(0);
                        Mark m1 = selm.ElementAt(1);
                        double fdx = m0.FoundPos.X - m1.FoundPos.X;
                        double fdy = m0.FoundPos.Y - m1.FoundPos.Y;
                        double mdx = m0.MapPos.X - m1.MapPos.X;
                        double mdy = m0.MapPos.Y - m1.MapPos.Y;
                        double e = Math.Sqrt(fdx * fdx + fdy * fdy) / Math.Sqrt(mdx * mdx + mdy * mdy);
                        double r = Math.Atan2(fdy, fdx) - Math.Atan2(mdy, mdx);
                        double cr = Math.Cos(r);
                        double sr = Math.Sin(r);
                        double d0x = m0.FoundPos.X - m0.MapPos.X;
                        double d0y = m0.FoundPos.Y - m0.MapPos.Y;

                        m_Map.MXX = m_Map.MYY = cr;
                        m_Map.MXY = -sr;
                        m_Map.MYX = sr;
                        m_Map.TX = d0x;
                        m_Map.TY = d0y;
                        m_Map.RX = m0.MapPos.X;
                        m_Map.RY = m0.MapPos.Y;
                    }
                    return;

                default:
                    {
                        double[] inDX = new double[selm.Count()];
                        double[] inDY = new double[selm.Count()];
                        double[] inX = new double[selm.Count()];
                        double[] inY = new double[selm.Count()];
                        int i;
                        for (i = 0; i < selm.Count(); i++)
                        {
                            Mark s = selm.ElementAt(i);
                            inX[i] = s.MapPos.X - selm.First().MapPos.X;
                            inY[i] = s.MapPos.Y - selm.First().MapPos.Y;
                            inDX[i] = s.FoundPos.X - s.MapPos.X;
                            inDY[i] = s.FoundPos.Y - s.MapPos.Y;
                        }
                        double[] outpar = new double[7];
                        if (NumericalTools.Fitting.Affine(inDX, inDY, inX, inY, ref outpar) == NumericalTools.ComputationResult.OK)
                        {
                            m_Map.MXX = (outpar[0] + 1.0);
                            m_Map.MXY = outpar[1];
                            m_Map.MYX = outpar[2];
                            m_Map.MYY = (outpar[3] + 1.0);
                            m_Map.TX = outpar[4];
                            m_Map.TY = outpar[5];
                            m_Map.RX = selm.First().MapPos.X;
                            m_Map.RY = selm.First().MapPos.Y;
                        }
                        else
                        {
                            m_Map.MXX = m_Map.MYY = 1.0;
                            m_Map.MXY = m_Map.MYX = 0.0;
                            m_Map.TX = selm.First().FoundPos.X - selm.First().MapPos.X;
                            m_Map.TY = selm.First().FoundPos.Y - selm.First().MapPos.Y;
                        }
                    }
                    return;
            }
        }

        private void OnLoad(object sender, EventArgs e)
        {
            if (m_PlateDesc != null)
            {
                txtMarkString.Text = m_PlateDesc.MapInitString;
                txtMarkString.Enabled = false;
            }
            else txtMarkString.Enabled = true;
        }
    }
}
