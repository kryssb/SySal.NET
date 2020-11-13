using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{
    public partial class CSToBrickForm : Form
    {
        class TrackObject
        {
            public SySal.Tracking.MIPEmulsionTrackInfo Info;
        }

        class VetoTrack : TrackObject
        {
            public long IdProcOp;
            public long IdEvent;
            public int Track;

            public override string ToString()
            {
                return "VT " + IdProcOp + " Ev " + IdEvent + " Tk " + Track + " PX " + Info.Intercept.X + " PY " + Info.Intercept.Y + " SX " + Info.Slope.X + " SY " + Info.Slope.Y;
            }
        }

        VetoTrack[] VetoTracks;

        class CSBrickCand : TrackObject
        {
            public long IdCand;            
            public SySal.BasicTypes.Vector2 DPos;
            public SySal.BasicTypes.Vector2 DSlope;
            public long IdZone;
            public int IdBase;
            public int IdUp;
            public int IdDown;            

            public override string ToString()
            {
                return "BK " + IdCand + " pl " + Info.Field + " G " + Info.Count + " PX " + Info.Intercept.X + " PY " + Info.Intercept.Y + " SX " + Info.Slope.X + " SY " + Info.Slope.Y + " S " + Info.Sigma + "\r\n" +
                    "DPX " + DPos.X + " DPY " + DPos.Y + " DSX " + DSlope.X + " DSY " + DSlope.Y;
            }

            public string SetPathText
            {
                get
                {
                    return IdCand + " " + IdZone + " " + IdBase + " " + IdUp + " " + IdDown;
                }
            }
        }

        CSBrickCand[] CSBrickCands = new CSBrickCand[0];

        CSBrickCand[] CSBrickNotFound = new CSBrickCand[0];

        class CSCand : TrackObject
        {
            public long IdCand;            

            public override string ToString()
            {
                return "CS " + IdCand + " G " + Info.Count + " PX " + Info.Intercept.X + " PY " + Info.Intercept.Y + " SX " + Info.Slope.X + " SY " + Info.Slope.Y;
            }
        }

        CSCand[] CSCands;

        class TTPred : TrackObject
        {
            public int Track;
            public long EventId;
            public string Type;
            public double Momentum;
            public int PDGId;            

            public override string ToString()
            {
                return "TT " + EventId + " TK " + Track + " Type " + Type + " Momentum " + Momentum + " PDGId " + PDGId + " PX " + Info.Intercept.X + " PY " + Info.Intercept.Y + " SX " + Info.Slope.X + " SY " + Info.Slope.Y;
            }
        }

        TTPred[] TTPreds;

        class Plate
        {
            public int Id;
            public double Z;

            public override string ToString()
            {
                return "PL " + Id + " Z " + Z;
            }
        }

        Plate[] Plates;

        int BrickId;

        int CSId;

        string ResultFile;

        double MinZ = 4550.0;

        const string DBResultsPrefix = @"db:\";

        bool CanSetPath = false;

        public CSToBrickForm(int bkid, string resultfile)
        {
            string milestone = "starting";            
            try
            {
                BrickId = bkid;
                ResultFile = resultfile;

                SySal.OperaDb.OperaDbConnection conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();

                int i;

                milestone = "getting CS";
                CSId = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select max(id) as maxid from tb_eventbricks where mod(id, 1000000) = " + (BrickId % 1000000) + " and id >= 3000000", conn).ExecuteScalar());

                System.Data.DataSet pl_ds = new System.Data.DataSet();
                milestone = "getting plates";
                new SySal.OperaDb.OperaDbDataAdapter("select id, z from tb_plates where id_eventbrick = " + BrickId + " order by z desc", conn).Fill(pl_ds);
                Plates = new Plate[pl_ds.Tables[0].Rows.Count];
                for (i = 0; i < pl_ds.Tables[0].Rows.Count; i++)
                {
                    Plates[i] = new Plate();
                    Plates[i].Id = SySal.OperaDb.Convert.ToInt32(pl_ds.Tables[0].Rows[i][0]);
                    Plates[i].Z = SySal.OperaDb.Convert.ToDouble(pl_ds.Tables[0].Rows[i][1]);
                }
                MinZ = Plates[0].Z;
                System.Data.DataSet pr_ds = new System.Data.DataSet();
                milestone = "getting TT Predictions";
                new SySal.OperaDb.OperaDbDataAdapter("select idev, track, posx * 1000 as px, posy * 1000 as py, slopex, slopey, tktyp, pdgid, momentum from tb_predicted_tracks " +
                    "inner join (select id_event as idev, track as idtk, pdgid, type as tktyp, momentum from tv_predtrack_brick_assoc where id_cs_eventbrick = " + CSId + ")" +
                    "on (id_event = idev and track = idtk)", conn).Fill(pr_ds);
                TTPreds = new TTPred[pr_ds.Tables[0].Rows.Count];
                for (i = 0; i < pr_ds.Tables[0].Rows.Count; i++)
                {
                    System.Data.DataRow dr = pr_ds.Tables[0].Rows[i];
                    TTPred t = new TTPred();
                    t.EventId = SySal.OperaDb.Convert.ToInt64(dr[0]);
                    t.Track = SySal.OperaDb.Convert.ToInt32(dr[1]);
                    t.Info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    t.Info.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[2]);
                    t.Info.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[3]);
                    t.Info.Intercept.Z = 4850.0;
                    t.Info.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[4]);
                    t.Info.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[5]);
                    t.Info.Intercept.X += (Plates[0].Z - t.Info.Intercept.Z) * t.Info.Slope.X;
                    t.Info.Intercept.Y += (Plates[0].Z - t.Info.Intercept.Z) * t.Info.Slope.Y;
                    t.Info.Intercept.Z = Plates[0].Z;
                    t.Info.TopZ = 4850.0;
                    t.Info.BottomZ = t.Info.Intercept.Z;
                    t.Type = dr[6].ToString();
                    t.PDGId = SySal.OperaDb.Convert.ToInt32(dr[7]);
                    t.Momentum = SySal.OperaDb.Convert.ToDouble(dr[8]);
                    TTPreds[i] = t;
                }
                System.Data.DataSet cn_ds = new System.Data.DataSet();
                milestone = "getting CS candidates";
                new SySal.OperaDb.OperaDbDataAdapter("select idcand, sum(grains) as grains, decode(sum(id_plate * decode(rnum,1,1,0)),1,4850,2,4550) as z, sum(posx * decode(rnum,1,1,0)) as posx, sum(posy * decode(rnum,1,1,0)) as posy, sum(slopex * decode(rnum,1,1,0)) as slopex, sum(slopey * decode(rnum,1,1,0)) as slopey from " +
                    "(select idcand, id_plate, grains, posx, posy, slopex, slopey, row_number() over (partition by idcand order by grains desc, id_plate desc) as rnum from vw_local_cs_candidates where id_cs_eventbrick = " + CSId + ")" +
                    "group by idcand", conn).Fill(cn_ds);
                CSCands = new CSCand[cn_ds.Tables[0].Rows.Count];
                for (i = 0; i < cn_ds.Tables[0].Rows.Count; i++)
                {
                    System.Data.DataRow dr = cn_ds.Tables[0].Rows[i];
                    CSCand c = new CSCand();
                    c.IdCand = SySal.OperaDb.Convert.ToInt64(dr[0]);
                    c.Info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    c.Info.Count = SySal.OperaDb.Convert.ToUInt16(dr[1]);
                    c.Info.Intercept.Z = SySal.OperaDb.Convert.ToDouble(dr[2]);
                    c.Info.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[3]);
                    c.Info.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[4]);
                    c.Info.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[5]);
                    c.Info.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[6]);
                    c.Info.Intercept.X += (Plates[0].Z - c.Info.Intercept.Z) * c.Info.Slope.X;
                    c.Info.Intercept.Y += (Plates[0].Z - c.Info.Intercept.Z) * c.Info.Slope.Y;
                    c.Info.TopZ = c.Info.Intercept.Z;
                    c.Info.BottomZ = c.Info.Intercept.Z = Plates[0].Z;
                    CSCands[i] = c;
                }
                System.Data.DataSet vt_ds = new System.Data.DataSet();
                milestone = "getting vetoed tracks";
                new SySal.OperaDb.OperaDbDataAdapter("select id_processoperation, id_event, track, posx, posy, slopex, slopey from tv_veto_tracks where id_cs_eventbrick = " + CSId, conn).Fill(vt_ds);
                VetoTracks = new VetoTrack[vt_ds.Tables[0].Rows.Count];
                for (i = 0; i < vt_ds.Tables[0].Rows.Count; i++)
                {
                    System.Data.DataRow dr = vt_ds.Tables[0].Rows[i];
                    VetoTrack c = new VetoTrack();
                    c.IdProcOp = SySal.OperaDb.Convert.ToInt64(dr[0]);
                    c.IdEvent = SySal.OperaDb.Convert.ToInt64(dr[1]);
                    c.Track = SySal.OperaDb.Convert.ToInt32(dr[2]);
                    c.Info = new SySal.Tracking.MIPEmulsionTrackInfo();                    
                    c.Info.Intercept.Z = 4550.0;
                    c.Info.Intercept.X = SySal.OperaDb.Convert.ToDouble(dr[3]);
                    c.Info.Intercept.Y = SySal.OperaDb.Convert.ToDouble(dr[4]);
                    c.Info.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[5]);
                    c.Info.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[6]);
                    c.Info.Intercept.X += (Plates[0].Z - c.Info.Intercept.Z) * c.Info.Slope.X;
                    c.Info.Intercept.Y += (Plates[0].Z - c.Info.Intercept.Z) * c.Info.Slope.Y;
                    c.Info.TopZ = c.Info.Intercept.Z;
                    c.Info.BottomZ = c.Info.Intercept.Z = Plates[0].Z;
                    VetoTracks[i] = c;
                }                
                
                System.Collections.ArrayList tkarr = new System.Collections.ArrayList();
                System.Collections.ArrayList seekarr = new System.Collections.ArrayList();
                if (ResultFile != null && ResultFile.ToLower().StartsWith(DBResultsPrefix) == false)
                {
                    milestone = "opening result file";
                    System.Text.RegularExpressions.Regex f_cand_rx = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+(\d+)\s+\d+\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");
                    System.Text.RegularExpressions.Match m_cand_rx = null;
                    System.IO.StreamReader r = new System.IO.StreamReader(ResultFile);
                    string line;
                    long cand = 0;
                    while ((line = r.ReadLine()) != null)
                    {
                        m_cand_rx = f_cand_rx.Match(line);
                        if (m_cand_rx.Success)
                        {
                            ++cand/* = Convert.ToInt64(m_cand_rx.Groups[1].Value)*/;
                            int plate = Convert.ToInt32(m_cand_rx.Groups[2].Value);
                            double z = 0.0;
                            foreach (System.Data.DataRow dr in pl_ds.Tables[0].Rows)
                                if (SySal.OperaDb.Convert.ToInt32(dr[0]) == plate)
                                {
                                    z = SySal.OperaDb.Convert.ToDouble(dr[1]);
                                    break;
                                }
                            if (z < MinZ) MinZ = z;
                            SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                            info.Field = (uint)plate;
                            info.Count = Convert.ToUInt16(m_cand_rx.Groups[3].Value);
                            info.AreaSum = Convert.ToUInt32(m_cand_rx.Groups[4].Value);
                            info.Intercept.X = Convert.ToDouble(m_cand_rx.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                            info.Intercept.Y = Convert.ToDouble(m_cand_rx.Groups[6].Value, System.Globalization.CultureInfo.InvariantCulture);
                            info.Intercept.Z = z;
                            info.Slope.X = Convert.ToDouble(m_cand_rx.Groups[7].Value, System.Globalization.CultureInfo.InvariantCulture);
                            info.Slope.Y = Convert.ToDouble(m_cand_rx.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture);
                            info.Sigma = Convert.ToDouble(m_cand_rx.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                            info.TopZ = info.Intercept.Z + 45.0;
                            info.BottomZ = info.Intercept.Z - 255.0;
                            SySal.BasicTypes.Vector2 dpos = new SySal.BasicTypes.Vector2();
                            dpos.X = Convert.ToDouble(m_cand_rx.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                            dpos.Y = Convert.ToDouble(m_cand_rx.Groups[11].Value, System.Globalization.CultureInfo.InvariantCulture);
                            SySal.BasicTypes.Vector2 dslope = new SySal.BasicTypes.Vector2();
                            dslope.X = Convert.ToDouble(m_cand_rx.Groups[12].Value, System.Globalization.CultureInfo.InvariantCulture);
                            dslope.Y = Convert.ToDouble(m_cand_rx.Groups[13].Value, System.Globalization.CultureInfo.InvariantCulture);
                            CSBrickCand csbkcand = new CSBrickCand();
                            csbkcand.IdCand = cand;
                            csbkcand.Info = info;
                            csbkcand.DPos = dpos;
                            csbkcand.DSlope = dslope;
                            tkarr.Add(csbkcand);
                        }
                    }
                    r.Close();
                }
                else
                {
                    uint mingrains = 0;
                    bool usemicrotracks = false;                    
                    if (ResultFile != null && ResultFile.ToLower().StartsWith(DBResultsPrefix))
                        ProcOpId = SySal.OperaDb.Convert.ToInt64(ResultFile.Substring(DBResultsPrefix.Length));
                    else
                    {
                        CSBrickSelProcOp bks = new CSBrickSelProcOp();
                        System.Data.DataSet op_ds = new System.Data.DataSet();
                        milestone = "getting CS-Brick connection operations";
                        new SySal.OperaDb.OperaDbDataAdapter(
                            "select idop, name, finishtime, success from tb_machines inner join " +
                            "(select id as idop, id_machine, finishtime, success from tb_proc_operations where id_eventbrick = " + BrickId + " and id_parent_operation is null and id_programsettings in " +
                            " (select id from tb_programsettings where instr(upper(description),'CS-BRICK') > 0)" +
                            ") on (id_machine = id)",                            
                            conn).Fill(op_ds);
                        bks.Rows = op_ds.Tables[0].Rows;
                        if (bks.ShowDialog() == DialogResult.OK)
                        {
                            ProcOpId = (long)bks.ProcOp;
                            mingrains = bks.MinGrains;
                            usemicrotracks = bks.UseMicrotracks;
                        }
                    }
                    if (ProcOpId > 0)
                    {                        
                        milestone = "checking if path can be set";
                        CanSetPath = (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT count(*) FROM TB_PROC_OPERATIONS WHERE SUCCESS = 'R' AND ID = " + ProcOpId, conn).ExecuteScalar()) > 0);
                        milestone = "getting search zones";                        
                        System.Data.DataSet zs_ds = new System.Data.DataSet();
                        new SySal.OperaDb.OperaDbDataAdapter(
                            "select idb, id_path, path, id_plate, ppx, ppy, psx, psy, 0 as grains, 0 as areasum, 0 as fpx, 0 as fpy, 0 as fsx, 0 as fsy, 0 as sigma, idz, 0 as idbase, 0 as idup, 0 as iddown from " +
                            "(select idb, id_path, path, idpl as id_plate, ppx, ppy, psx, psy, id as idz from tb_zones right join " +
                            " (select idb, id_path, path, id_plate as idpl, posx as ppx, posy as ppy, slopex as psx, slopey as psy from tb_scanback_predictions inner join " +
                            "  (select id_eventbrick as idb, id as idpath, path from tb_scanback_paths where id_eventbrick = " + BrickId + " and id_processoperation = " + ProcOpId + " and id_fork_path is null) " +
                            "  on (id_eventbrick = idb and id_path = idpath) " +
                            " ) on (id_eventbrick = idb and tb_zones.id_plate = idpl and series = id_path) " +
                            ")",
                            conn).Fill(zs_ds);
                        foreach (System.Data.DataRow zs_dr in zs_ds.Tables[0].Rows)
                        {
                            CSBrickCand csbkcand = new CSBrickCand();
                            csbkcand.IdCand = Convert.ToInt64(zs_dr[2]);
                            csbkcand.Info = new SySal.Tracking.MIPEmulsionTrackInfo();
                            csbkcand.Info.Field = Convert.ToUInt32(zs_dr[3]);
                            csbkcand.Info.AreaSum = Convert.ToUInt32(zs_dr[9]);
                            csbkcand.Info.Count = Convert.ToUInt16(zs_dr[8]);
                            csbkcand.Info.Intercept.X = Math.Round(Convert.ToDouble(zs_dr[10]), 1);
                            csbkcand.Info.Intercept.Y = Math.Round(Convert.ToDouble(zs_dr[11]), 1);

                            double z = 0.0;
                            foreach (System.Data.DataRow dr in pl_ds.Tables[0].Rows)
                                if (SySal.OperaDb.Convert.ToUInt32(dr[0]) == csbkcand.Info.Field)
                                {
                                    z = SySal.OperaDb.Convert.ToDouble(dr[1]);
                                    break;
                                }
                            if (z < MinZ) MinZ = z;

                            csbkcand.Info.Intercept.Z = z;
                            csbkcand.Info.Slope.X = Math.Round(Convert.ToDouble(zs_dr[12]), 4);
                            csbkcand.Info.Slope.Y = Math.Round(Convert.ToDouble(zs_dr[13]), 4);
                            csbkcand.Info.Sigma = Math.Round(Convert.ToDouble(zs_dr[14]), 4);
                            csbkcand.IdZone = Convert.ToInt64(zs_dr[15]);
                            csbkcand.IdBase = Convert.ToInt32(zs_dr[16]);
                            csbkcand.IdUp = Convert.ToInt32(zs_dr[17]);
                            csbkcand.IdDown = Convert.ToInt32(zs_dr[18]);
                            csbkcand.Info.BottomZ = z - 255.0;
                            csbkcand.Info.TopZ = z + 45.0;
                            csbkcand.DPos.X = Math.Round(Convert.ToDouble(zs_dr[4]) - csbkcand.Info.Intercept.X, 1);
                            csbkcand.DPos.Y = Math.Round(Convert.ToDouble(zs_dr[5]) - csbkcand.Info.Intercept.Y, 1);
                            csbkcand.DSlope.X = Math.Round(Convert.ToDouble(zs_dr[6]) - csbkcand.Info.Slope.X, 4);
                            csbkcand.DSlope.Y = Math.Round(Convert.ToDouble(zs_dr[7]) - csbkcand.Info.Slope.Y, 4);
                            seekarr.Add(csbkcand);
                        }
                        milestone = "getting scanning results";
                        System.Data.DataSet rs_ds = new System.Data.DataSet();
                        /*
                        new SySal.OperaDb.OperaDbDataAdapter(
                            "select idb, id_path, path, id_plate, ppx, ppy, psx, psy, grains, areasum, posx as fpx, posy as fpy, slopex as fsx, slopey as fsy, sigma from " +
                            "(select idb, id_path, path, id_plate, posx as ppx, posy as ppy, slopex as psx, slopey as psy, id_zone as idz, id_candidate from tb_scanback_predictions inner join " +
                            " (select id_eventbrick as idb, id as idpath, path from tb_scanback_paths where id_eventbrick = " + BrickId + " and id_processoperation = " + procopid + ") " +
                            " on (id_eventbrick = idb and id_path = idpath and id_candidate is not null) " +
                            ") inner join tb_mipbasetracks on (id_eventbrick = idb and id_zone = idz and id = id_candidate)",
                            conn).Fill(rs_ds);*/
                        /*
                        new SySal.OperaDb.OperaDbDataAdapter(
                            "select idb, id_path, path, id_plate, ppx, ppy, psx, psy, grains, areasum, posx as fpx, posy as fpy, slopex as fsx, slopey as fsy, sigma, idz, id as idbase, 0 as idup, 0 as iddown from " +
                            "(select idb, id_path, path, id_plate, posx as ppx, posy as ppy, slopex as psx, slopey as psy, id_zone as idz, id_candidate from tb_scanback_predictions inner join " +
                            " (select id_eventbrick as idb, id as idpath, path from tb_scanback_paths where id_eventbrick = " + BrickId + " and id_processoperation = " + ProcOpId + " and id_fork_path is null) " +
                            " on (id_eventbrick = idb and id_path = idpath) " +
                            ") inner join tb_mipbasetracks on (id_eventbrick = idb and id_zone = idz and grains >= " + mingrains + ")" + (usemicrotracks ?  
                            "\r\n" +
                            "union\r\n" +
                            "\r\n" +
                            "select idb, id_path, path, id_plate, ppx, ppy, psx, psy, grains, areasum, posx + (side - 1) * 205 * slopex as fpx, posy + (side - 1) * 205 * slopey as fpy, slopex as fsx, slopey as fsy, sigma, idz, 0 as idbase, decode(side,2,id,0) as idup, decode(side,1,id,0) as iddown from " +
                            "(select idb, id_path, path, id_plate, posx as ppx, posy as ppy, slopex as psx, slopey as psy, id_zone as idz, id_candidate from tb_scanback_predictions inner join " +
                            " (select id_eventbrick as idb, id as idpath, path from tb_scanback_paths where id_eventbrick = " + BrickId + " and id_processoperation = " + ProcOpId + " and id_fork_path is null) " +
                            " on (id_eventbrick = idb and id_path = idpath) " +
                            ") inner join tb_mipmicrotracks on (id_eventbrick = idb and id_zone = idz and grains >= " + mingrains + ")" : ""),
                            conn).Fill(rs_ds);
                         */
                        new SySal.OperaDb.OperaDbDataAdapter(
                            "select idb, id_path, path, id_plate, ppx, ppy, psx, psy, grains, areasum, posx as fpx, posy as fpy, slopex as fsx, slopey as fsy, sigma, idz, id as idbase, 0 as idup, 0 as iddown from " +
                            "(select idb, id_path, path, idpl as id_plate, ppx, ppy, psx, psy, id as idz from tb_zones right join " +
                            " (select idb, id_path, path, id_plate as idpl, posx as ppx, posy as ppy, slopex as psx, slopey as psy from tb_scanback_predictions inner join " +
                            "  (select id_eventbrick as idb, id as idpath, path from tb_scanback_paths where id_eventbrick = " + BrickId + " and id_processoperation = " + ProcOpId + " and id_fork_path is null) " +
                            "  on (id_eventbrick = idb and id_path = idpath) " +
                            " ) on (id_eventbrick = idb and tb_zones.id_plate = idpl and series = id_path) " +
                            ") inner join tb_mipbasetracks on (id_eventbrick = idb and id_zone = idz and grains >= " + mingrains + ")" + (usemicrotracks ?
                            "\r\n" +
                            "union\r\n" +
                            "\r\n" +
                            "select idb, id_path, path, id_plate, ppx, ppy, psx, psy, grains, areasum, posx + (side - 1) * 205 * slopex as fpx, posy + (side - 1) * 205 * slopey as fpy, slopex as fsx, slopey as fsy, sigma, idz, 0 as idbase, decode(side,2,id,0) as idup, decode(side,1,id,0) as iddown from " +
                            "(select idb, id_path, path, idpl as id_plate, ppx, ppy, psx, psy, id as idz from tb_zones right join " +
                            " (select idb, id_path, path, id_plate as idpl, posx as ppx, posy as ppy, slopex as psx, slopey as psy from tb_scanback_predictions inner join " +
                            "  (select id_eventbrick as idb, id as idpath, path from tb_scanback_paths where id_eventbrick = " + BrickId + " and id_processoperation = " + ProcOpId + " and id_fork_path is null) " +
                            "  on (id_eventbrick = idb and id_path = idpath) " +
                            " ) on (id_eventbrick = idb and tb_zones.id_plate = idpl and series = id_path) " +
                            ") inner join tb_mipmicrotracks on (id_eventbrick = idb and id_zone = idz and grains >= " + mingrains + ")" : ""),
                            conn).Fill(rs_ds);
                        foreach (System.Data.DataRow rs_dr in rs_ds.Tables[0].Rows)
                        {
                            CSBrickCand csbkcand = new CSBrickCand();                            
                            csbkcand.IdCand = Convert.ToInt64(rs_dr[2]);
                            csbkcand.Info = new SySal.Tracking.MIPEmulsionTrackInfo();
                            csbkcand.Info.Field = Convert.ToUInt32(rs_dr[3]);
                            csbkcand.Info.AreaSum = Convert.ToUInt32(rs_dr[9]);
                            csbkcand.Info.Count = Convert.ToUInt16(rs_dr[8]);
                            csbkcand.Info.Intercept.X = Math.Round(Convert.ToDouble(rs_dr[10]), 1);
                            csbkcand.Info.Intercept.Y = Math.Round(Convert.ToDouble(rs_dr[11]), 1);

                            double z = 0.0;
                            foreach (System.Data.DataRow dr in pl_ds.Tables[0].Rows)
                                if (SySal.OperaDb.Convert.ToUInt32(dr[0]) == csbkcand.Info.Field)
                                {
                                    z = SySal.OperaDb.Convert.ToDouble(dr[1]);
                                    break;
                                }
                            if (z < MinZ) MinZ = z;

                            csbkcand.Info.Intercept.Z = z;
                            csbkcand.Info.Slope.X = Math.Round(Convert.ToDouble(rs_dr[12]), 4);
                            csbkcand.Info.Slope.Y = Math.Round(Convert.ToDouble(rs_dr[13]), 4);
                            csbkcand.Info.Sigma = Math.Round(Convert.ToDouble(rs_dr[14]), 4);
                            csbkcand.IdZone = Convert.ToInt64(rs_dr[15]);
                            csbkcand.IdBase = Convert.ToInt32(rs_dr[16]);
                            csbkcand.IdUp = Convert.ToInt32(rs_dr[17]);
                            csbkcand.IdDown = Convert.ToInt32(rs_dr[18]);                            
                            csbkcand.Info.BottomZ = z - 255.0;
                            csbkcand.Info.TopZ = z + 45.0;
                            csbkcand.DPos.X = Math.Round(Convert.ToDouble(rs_dr[4]) - csbkcand.Info.Intercept.X, 1);
                            csbkcand.DPos.Y = Math.Round(Convert.ToDouble(rs_dr[5]) - csbkcand.Info.Intercept.Y, 1);
                            csbkcand.DSlope.X = Math.Round(Convert.ToDouble(rs_dr[6]) - csbkcand.Info.Slope.X, 4);
                            csbkcand.DSlope.Y = Math.Round(Convert.ToDouble(rs_dr[7]) - csbkcand.Info.Slope.Y, 4);
                            tkarr.Add(csbkcand);
                        }
                    }
                }
                CSBrickCands = (CSBrickCand[])tkarr.ToArray(typeof(CSBrickCand));
                CSBrickNotFound = (CSBrickCand[])seekarr.ToArray(typeof(CSBrickCand));
                conn.Close();
                milestone = "exiting";
            }
            catch (Exception x)
            {
                MessageBox.Show("Error while " + milestone + ":\r\n" + x.ToString(), "Initialization Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }           
            InitializeComponent();
        }

        private void btnXY_Click(object sender, EventArgs e)
        {
            double x = 0, y = 0, z = 0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(0, 0, 1, 0, -1, 0);            
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();
        }

        private void btnXZ_Click(object sender, EventArgs e)
        {
            double x = 0, y = 0, z = 0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(0, -1, 0, 0, 0, -1);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();            
        }

        private void btnYZ_Click(object sender, EventArgs e)
        {
            double x = 0, y = 0, z = 0;
            gdiDisplay1.GetCameraSpotting(ref x, ref y, ref z);
            gdiDisplay1.SetCameraOrientation(1, 0, 0, 0, 0, -1);
            gdiDisplay1.SetCameraSpotting(x, y, z);
            gdiDisplay1.Render();           
        }

        private void btnZoomIn_Click(object sender, EventArgs e)
        {
            gdiDisplay1.Zoom = gdiDisplay1.Zoom * 1.1;            
            gdiDisplay1.Render();            
        }

        private void btnZoomOut_Click(object sender, EventArgs e)
        {
            gdiDisplay1.Zoom = gdiDisplay1.Zoom / 1.1;
            gdiDisplay1.Render();            
        }

        private void btnSetFocus_Click(object sender, EventArgs e)
        {
            gdiDisplay1.NextClickSetsCenter = true;
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Select file to dump plot";
            sdlg.Filter = "3D XML (*.x3l)|*.x3l|Portable Network Graphics (*.png)|*.png|Graphics Interexchange Format (*.gif)|*.gif|Joint Photographics Experts Group (*.jpg)|*.jpg|Windows Bitmap (*.bmp)|*.bmp";
            if (sdlg.ShowDialog() == DialogResult.OK)
                try
                {
                    gdiDisplay1.Save(sdlg.FileName);
                    MessageBox.Show("File written.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception x)
                {
                    MessageBox.Show("Can't save file:\r\n" + x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
        }

        public int MinGrains = 20;

        private void OnMinGrainsLeave(object sender, EventArgs e)
        {
            try
            {
                MinGrains = Convert.ToInt32(txtMinGrains.Text);
                ApplySelections();
            }
            catch (Exception)
            {
                txtMinGrains.Text = MinGrains.ToString();
                txtMinGrains.Focus();
            }
        }

        public double MaxSigma = 0.8;

        private void OnMaxSigmaLeave(object sender, EventArgs e)
        {
            try
            {
                MaxSigma = Convert.ToDouble(txtMaxSigma.Text, System.Globalization.CultureInfo.InvariantCulture);
                ApplySelections();
            }
            catch (Exception)
            {
                txtMaxSigma.Text = MaxSigma.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtMaxSigma.Focus();
            }
        }

        public double MaxDeltaPos = 300.0;

        private void OnMaxDeltaPosLeave(object sender, EventArgs e)
        {
            try
            {
                MaxDeltaPos = Convert.ToDouble(txtMaxDeltaPos.Text, System.Globalization.CultureInfo.InvariantCulture);
                ApplySelections();
            }
            catch (Exception)
            {
                txtMaxDeltaPos.Text = MaxDeltaPos.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtMaxDeltaPos.Focus();
            }
        }

        public double MaxDeltaSlope = 0.04;

        private void OnMaxDeltaSlopeLeave(object sender, EventArgs e)
        {
            try
            {
                MaxDeltaSlope = Convert.ToDouble(txtMaxDeltaSlope.Text, System.Globalization.CultureInfo.InvariantCulture);
                ApplySelections();
            }
            catch (Exception)
            {
                txtMaxDeltaSlope.Text = MaxDeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txtMaxDeltaSlope.Focus();
            }
        }

        private void btnRemoveTrack_Click(object sender, EventArgs e)
        {
            foreach (ListViewItem lvi in lvAcceptedTracks.SelectedItems)
            {
                lvi.Remove();
                gdiDisplay1.Highlight(lvi.Tag, false);

                if (VF != null) gdiDisplay1.DeleteWithOwner(VF);
                txtVtxX.Text = txtVtxY.Text = txtVtxZ.Text = "";
                txtVtxDownstreamPlate.Text = txtVtxDepth.Text = "";
                lvIPs.Items.Clear();
            }
        }

        private void btnExportSBInit_Click(object sender, EventArgs e)
        {
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Select file to save SB initialization information";
            sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            if (sdlg.ShowDialog() == DialogResult.OK)
                try
                {
                    string entries = "";
                    foreach (ListViewItem lvi in lvAcceptedTracks.Items)
                    {
                        if (entries.Length > 0) entries += "\r\n";
                        entries += lvi.SubItems[0].Text + " " + lvi.SubItems[1].Text + " " +
                            lvi.SubItems[2].Text + " " + lvi.SubItems[3].Text + " " +
                            lvi.SubItems[4].Text + " " + lvi.SubItems[5].Text;
                    }
                    System.IO.File.WriteAllText(sdlg.FileName, entries);
                    MessageBox.Show("File written.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception x)
                {
                    MessageBox.Show("Can't save file:\r\n" + x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
        }

        public int IdBrick;

        public long ProcOpId = 0;

        public string P3DFile;

        private void OnLoad(object sender, EventArgs e)
        {
            foreach (Plate p in Plates)
                cmbPlates.Items.Add(p.Id);
            cmbPlates.SelectedIndex = 0;
            txtMinGrains.Text = MinGrains.ToString();
            txtMaxSigma.Text = MaxSigma.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtMaxDeltaPos.Text = MaxDeltaPos.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtMaxDeltaSlope.Text = MaxDeltaSlope.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDownstreamPlates.Text = TSDownstreamPlates.ToString();
            txtUpstreamPlates.Text = TSUpstreamPlates.ToString();
            txtTSWidth.Text = TSWidth.ToString();
            txtTSHeight.Text = TSHeight.ToString();
            gdiDisplay1.DoubleClickSelect = new GDI3D.Control.SelectObject(SelectTrack);
            gdiDisplay1.AutoRender = false;

            foreach (TTPred t in TTPreds)
                gdiDisplay1.Add(new GDI3D.Control.Line(
                    t.Info.Intercept.X + (t.Info.TopZ - t.Info.Intercept.Z) * t.Info.Slope.X,
                    t.Info.Intercept.Y + (t.Info.TopZ - t.Info.Intercept.Z) * t.Info.Slope.Y,
                    t.Info.TopZ,
                    t.Info.Intercept.X + (MinZ - t.Info.Intercept.Z) * t.Info.Slope.X,
                    t.Info.Intercept.Y + (MinZ - t.Info.Intercept.Z) * t.Info.Slope.Y,
                    MinZ, t, 255, 128, 128));

            foreach (CSCand t in CSCands)
                gdiDisplay1.Add(new GDI3D.Control.Line(
                    t.Info.Intercept.X + (t.Info.TopZ - t.Info.Intercept.Z) * t.Info.Slope.X,
                    t.Info.Intercept.Y + (t.Info.TopZ - t.Info.Intercept.Z) * t.Info.Slope.Y,
                    t.Info.TopZ,
                    t.Info.Intercept.X + (MinZ - t.Info.Intercept.Z) * t.Info.Slope.X,
                    t.Info.Intercept.Y + (MinZ - t.Info.Intercept.Z) * t.Info.Slope.Y,
                    MinZ, t, 255, 128, 255));

            foreach (VetoTrack t in VetoTracks)
                gdiDisplay1.Add(new GDI3D.Control.Line(
                    t.Info.Intercept.X + (t.Info.TopZ - t.Info.Intercept.Z) * t.Info.Slope.X,
                    t.Info.Intercept.Y + (t.Info.TopZ - t.Info.Intercept.Z) * t.Info.Slope.Y,
                    t.Info.TopZ,
                    t.Info.Intercept.X + (MinZ - t.Info.Intercept.Z) * t.Info.Slope.X,
                    t.Info.Intercept.Y + (MinZ - t.Info.Intercept.Z) * t.Info.Slope.Y,
                    MinZ, t, 255, 255, 128));

            gdiDisplay1.Add(new GDI3D.Control.Line(0, 0, Plates[0].Z, 125000, 0, Plates[0].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(125000, 0, Plates[0].Z, 125000, 100000, Plates[0].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(125000, 100000, Plates[0].Z, 0, 100000, Plates[0].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(0, 100000, Plates[0].Z, 0, 0, Plates[0].Z, null, 64, 64, 192));

            gdiDisplay1.Add(new GDI3D.Control.Line(0, 0, Plates[Plates.Length - 1].Z, 125000, 0, Plates[Plates.Length - 1].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(125000, 0, Plates[Plates.Length - 1].Z, 125000, 100000, Plates[Plates.Length - 1].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(125000, 100000, Plates[Plates.Length - 1].Z, 0, 100000, Plates[Plates.Length - 1].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(0, 100000, Plates[Plates.Length - 1].Z, 0, 0, Plates[Plates.Length - 1].Z, null, 64, 64, 192));

            gdiDisplay1.Add(new GDI3D.Control.Line(0, 0, Plates[0].Z, 0, 0, Plates[Plates.Length - 1].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(125000, 0, Plates[0].Z, 125000, 0, Plates[Plates.Length - 1].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(125000, 100000, Plates[0].Z, 125000, 100000, Plates[Plates.Length - 1].Z, null, 64, 64, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(0, 100000, Plates[0].Z, 0, 100000, Plates[Plates.Length - 1].Z, null, 64, 64, 192));

            gdiDisplay1.Add(new GDI3D.Control.Line(-10000, -10000, Plates[0].Z, 0, -10000, Plates[0].Z, "X", 192, 192, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(-10000, -10000, Plates[0].Z, -10000, 0, Plates[0].Z, "Y", 192, 192, 192));
            gdiDisplay1.Add(new GDI3D.Control.Line(-10000, -10000, Plates[0].Z, -10000, -10000, Plates[0].Z + 10000, "Z", 192, 192, 192));

            gdiDisplay1.SetCameraSpotting(62500, 50000, 0);
            gdiDisplay1.SetCameraOrientation(0, 0, 1, 0, -1, 0);
            gdiDisplay1.Distance = 1000000;
            gdiDisplay1.Infinity = true;

            btnSetPathsAndCSBrick.Enabled = CanSetPath;

            ApplySelections();
        }

        static int[][] color = new int[][]
        { 
            new int[] {192,192,255},
            new int[] {192,255,192},
            new int[] {192,255,192},
            new int[] {255,255,192}            
        };

        void ApplySelections()
        {
            gdiDisplay1.AutoRender = false;
            foreach (CSBrickCand csbc in CSBrickCands)
                gdiDisplay1.DeleteWithOwner(csbc);
            foreach (CSBrickCand csbc in CSBrickCands)
                if (csbc.Info.Count >= MinGrains && csbc.Info.Sigma <= MaxSigma && Math.Max(Math.Abs(csbc.DPos.X), Math.Abs(csbc.DPos.Y)) <= MaxDeltaPos && Math.Max(Math.Abs(csbc.DSlope.X), Math.Abs(csbc.DSlope.Y)) <= MaxDeltaSlope)
                    gdiDisplay1.Add(new GDI3D.Control.Line(
                        csbc.Info.Intercept.X + (csbc.Info.TopZ - csbc.Info.Intercept.Z) * csbc.Info.Slope.X,
                        csbc.Info.Intercept.Y + (csbc.Info.TopZ - csbc.Info.Intercept.Z) * csbc.Info.Slope.Y,
                        csbc.Info.TopZ,
                        csbc.Info.Intercept.X + (csbc.Info.BottomZ - csbc.Info.Intercept.Z) * csbc.Info.Slope.X,
                        csbc.Info.Intercept.Y + (csbc.Info.BottomZ - csbc.Info.Intercept.Z) * csbc.Info.Slope.Y,
                        csbc.Info.BottomZ,
                        csbc,
                        color[csbc.Info.Field % color.Length][0], color[csbc.Info.Field % color.Length][1], color[csbc.Info.Field % color.Length][2]));
            gdiDisplay1.AutoRender = true;
            gdiDisplay1.Render();            
        }

        object SelectedObj = null;

        void SelectTrack(object o)
        {
            SelectedObj = o;
            txtTrackInfo.Text = SelectedObj.ToString();
        }

        private void btnAcceptCand_Click(object sender, EventArgs e)
        {
            if (SelectedObj != null && (SelectedObj is CSBrickCand || SelectedObj is TTPred || SelectedObj is CSCand))
            {
                foreach (ListViewItem lvi in lvAcceptedTracks.Items)
                    if (lvi.Tag == SelectedObj)
                    {
                        MessageBox.Show("Track already added.", "Input warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }
                double z = 0.0;
                int plate = Convert.ToInt32(cmbPlates.SelectedItem);
                foreach (Plate p in Plates)
                    if (plate == p.Id) 
                    {
                        z = p.Z;
                        break;
                    }                
                ListViewItem nlv = null;
                SySal.Tracking.MIPEmulsionTrackInfo Info = null;
                if (SelectedObj is CSBrickCand)
                {
                    CSBrickCand csbc = (CSBrickCand)SelectedObj;
                    nlv = new ListViewItem(csbc.IdCand.ToString());
                    Info = csbc.Info;
                }
                else if (SelectedObj is TTPred)
                {
                    TTPred ttp = (TTPred)SelectedObj;
                    nlv = new ListViewItem(ttp.Track.ToString());
                    Info = ttp.Info;
                }
                else if (SelectedObj is CSCand)
                {
                    CSCand csc = (CSCand)SelectedObj;
                    nlv = new ListViewItem(csc.IdCand.ToString());
                    Info = csc.Info;
                }
                nlv.Tag = SelectedObj;
                nlv.SubItems.Add((Info.Intercept.X + (z - Info.Intercept.Z) * Info.Slope.X).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                nlv.SubItems.Add((Info.Intercept.Y + (z - Info.Intercept.Z) * Info.Slope.Y).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                nlv.SubItems.Add(Info.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                nlv.SubItems.Add(Info.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                nlv.SubItems.Add(plate.ToString());
                lvAcceptedTracks.Items.Add(nlv);

                if (VF != null) gdiDisplay1.DeleteWithOwner(VF);
                txtVtxX.Text = txtVtxY.Text = txtVtxZ.Text = "";
                txtVtxDownstreamPlate.Text = txtVtxDepth.Text = "";
                lvIPs.Items.Clear();
            }
        }

        private void OnProjPlateChanged(object sender, EventArgs e)
        {

        }

        private void OnSelAcceptTrackChanged(object sender, EventArgs e)
        {
            gdiDisplay1.AutoRender = false;
            foreach (ListViewItem lvi in lvAcceptedTracks.Items)
                gdiDisplay1.Highlight(lvi.Tag, false);
            foreach (ListViewItem lvi in lvAcceptedTracks.SelectedItems)
                gdiDisplay1.Highlight(lvi.Tag, true);
            gdiDisplay1.AutoRender = true;
            gdiDisplay1.Render();
        }

        int TSDownstreamPlates = 10;

        private void OnDownstreamPlatesLeave(object sender, EventArgs e)
        {
            try
            {
                TSDownstreamPlates = Convert.ToInt32(txtDownstreamPlates.Text);                
            }
            catch (Exception)
            {
                txtDownstreamPlates.Text = TSDownstreamPlates.ToString();
                txtDownstreamPlates.Focus();
            }       
        }

        int TSUpstreamPlates = 5;

        private void OnUpstreamPlatesLeave(object sender, EventArgs e)
        {
            try
            {
                TSUpstreamPlates = Convert.ToInt32(txtUpstreamPlates.Text);
            }
            catch (Exception)
            {
                txtUpstreamPlates.Text = TSUpstreamPlates.ToString();
                txtUpstreamPlates.Focus();
            }       
        }

        int TSWidth = 10000;

        private void OnTSWidthLeave(object sender, EventArgs e)
        {
            try
            {
                TSWidth = Convert.ToInt32(txtTSWidth.Text);
            }
            catch (Exception)
            {
                txtTSWidth.Text = TSWidth.ToString();
                txtTSWidth.Focus();
            }       
        }

        int TSHeight = 10000;

        private void OnTSHeightLeave(object sender, EventArgs e)
        {
            try
            {
                TSHeight = Convert.ToInt32(txtTSHeight.Text);
            }
            catch (Exception)
            {
                txtTSHeight.Text = TSHeight.ToString();
                txtTSHeight.Focus();
            }       
        }

        private void btnExportTSInit_Click(object sender, EventArgs e)
        {
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Select file to save TS initialization information";
            sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
            if (sdlg.ShowDialog() == DialogResult.OK)
                try
                {
                    string entries = "";
                    foreach (ListViewItem lvi in lvAcceptedTracks.Items)
                    {
                        if (entries.Length > 0) entries += "\r\n";
                        SySal.Tracking.MIPEmulsionTrackInfo info = ((TrackObject)lvi.Tag).Info;
                        int plate = Convert.ToInt32(lvi.SubItems[5].Text);
                        int i;
                        for (i = 0; i < Plates.Length && plate != Plates[i].Id; i++) ;
                        if (i >= Plates.Length)
                        {
                            MessageBox.Show("Invalid plate specification (" + plate + ")", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                            return;
                        }
                        entries += lvi.SubItems[0].Text + " " +
                            (info.Intercept.X - 0.5 * TSWidth).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " " +
                            (info.Intercept.X + 0.5 * TSWidth).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " " +
                            (info.Intercept.Y - 0.5 * TSHeight).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " " +
                            (info.Intercept.Y + 0.5 * TSHeight).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " " +
                            Plates[Math.Min(i + TSUpstreamPlates, Plates.Length - 1)].Id.ToString() + " " + 
                            Plates[Math.Max(i - TSDownstreamPlates, 0)].Id.ToString() + " " + 
                            plate;
                        if (chkTSUseSlope.Checked)
                            entries += " " + lvi.SubItems[3].Text + " " + lvi.SubItems[4].Text;
                        else
                            entries += " 0 0";
                    }
                    System.IO.File.WriteAllText(sdlg.FileName, entries);
                    MessageBox.Show("File written.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception x)
                {
                    MessageBox.Show("Can't save file:\r\n" + x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
        }

        SySal.TotalScan.VertexFit VF = null;

        private void btnVertex_Click(object sender, EventArgs e)
        {
            try
            {
                if (VF != null) gdiDisplay1.DeleteWithOwner(VF);                
                VF = null;
                txtVtxX.Text = txtVtxY.Text = txtVtxZ.Text = "";
                txtVtxDownstreamPlate.Text = txtVtxDepth.Text = "";
                lvIPs.Items.Clear();
                VF = new SySal.TotalScan.VertexFit();
                foreach (ListViewItem lvi in lvAcceptedTracks.Items)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo Info = ((TrackObject)lvi.Tag).Info;
                    SySal.TotalScan.VertexFit.TrackFit tf = new SySal.TotalScan.VertexFit.TrackFit();
                    tf.Id = new SySal.TotalScan.BaseTrackIndex(lvi.Index + 1);
                    tf.Field = Info.Field;
                    tf.Count = Info.Count;
                    tf.AreaSum = Info.AreaSum;
                    tf.Intercept = Info.Intercept;
                    tf.Slope = Info.Slope;
                    tf.Slope.Z = 1.0;
                    tf.Sigma = Info.Sigma;
                    tf.TopZ = Info.TopZ;
                    tf.BottomZ = Info.BottomZ;
                    tf.MaxZ = Info.BottomZ;
                    tf.MinZ = Plates[Plates.Length - 1].Z - 10000.0;                                        
                    tf.Weight = 1.0;
                    
                    VF.AddTrackFit(tf);
                }
                txtVtxX.Text = VF.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                txtVtxY.Text = VF.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                txtVtxZ.Text = VF.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                int i;
                for (i = Plates.Length - 1; i >= 0 && Plates[i].Z < VF.Z; i--) ;
                txtVtxDownstreamPlate.Text = Plates[i].Id.ToString();
                txtVtxDepth.Text = (Plates[i].Z - VF.Z).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                for (i = 0; i < VF.Count; i++)
                {
                    ListViewItem nvl = new ListViewItem((i + 1).ToString());
                    nvl.SubItems.Add(VF.TrackIP(VF.Track(i)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    try
                    {
                        nvl.SubItems.Add(VF.DisconnectedTrackIP(VF.Track(i).Id).ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    }
                    catch (Exception)
                    {
                        nvl.SubItems.Add("");
                    }
                    lvIPs.Items.Add(nvl);
                }
                gdiDisplay1.Add(new GDI3D.Control.Point(VF.X, VF.Y, VF.Z, VF, 255, 255, 255));
                for (i = 0; i < VF.Count; i++)
                {
                    SySal.TotalScan.VertexFit.TrackFit tf = VF.Track(i);
                    GDI3D.Control.Line ln = new GDI3D.Control.Line(
                        tf.Intercept.X + (tf.BottomZ - tf.Intercept.Z) * tf.Slope.X, tf.Intercept.Y + (tf.BottomZ - tf.Intercept.Z) * tf.Slope.Y, tf.BottomZ,
                        tf.Intercept.X + (VF.Z - tf.Intercept.Z) * tf.Slope.X, tf.Intercept.Y + (VF.Z - tf.Intercept.Z) * tf.Slope.Y, VF.Z,
                        VF, 255, 255, 255);
                    ln.Dashed = true;
                    gdiDisplay1.Add(ln);
                }
            }
            catch (Exception x)
            {
                MessageBox.Show("Can't make vertex:\r\n" + x.ToString(), "Vertex Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                VF = null;
            }
        }

        private void btnAddVertexPos_Click(object sender, EventArgs e)
        {
            if (VF != null)
            {
                ListViewItem nvl = new ListViewItem(txtVtxDownstreamPlate.Text);
                nvl.SubItems.Add(txtVtxX.Text);
                nvl.SubItems.Add(txtVtxY.Text);
                nvl.SubItems.Add("0");
                nvl.SubItems.Add("0");
                nvl.SubItems.Add(txtVtxDownstreamPlate.Text);
                SySal.Tracking.MIPEmulsionTrackInfo Info = new SySal.Tracking.MIPEmulsionTrackInfo();
                Info.Intercept.X = VF.X;
                Info.Intercept.Y = VF.Y;
                Info.Intercept.Z = VF.Z;
                Info.Slope.X = 0.0;
                Info.Slope.Y = 0.0;
                Info.Slope.Z = 1.0;
                TrackObject to = new TrackObject();
                to.Info = Info;
                nvl.Tag = to;
                lvAcceptedTracks.Items.Add(nvl);
            }
        }

        private void btnTrackFollowAutoStart_Click(object sender, EventArgs e)
        {
            if (lvAcceptedTracks.Items.Count <= 0) return;
            {
                EnqueueOpForm ef = new EnqueueOpForm();
                ef.BrickId = BrickId;
                EnqueueOpForm.TrackFollowInfo[] tk = new EnqueueOpForm.TrackFollowInfo[lvAcceptedTracks.Items.Count];
                int i;
                for (i = 0; i < lvAcceptedTracks.Items.Count; i++)
                {
                    tk[i].Id = i + 1;
                    tk[i].Plate = Convert.ToInt32(lvAcceptedTracks.Items[i].SubItems[5].Text);
                    tk[i].Position.X = Convert.ToDouble(lvAcceptedTracks.Items[i].SubItems[1].Text, System.Globalization.CultureInfo.InvariantCulture);
                    tk[i].Position.Y = Convert.ToDouble(lvAcceptedTracks.Items[i].SubItems[2].Text, System.Globalization.CultureInfo.InvariantCulture);
                    tk[i].Slope.X = Convert.ToDouble(lvAcceptedTracks.Items[i].SubItems[3].Text, System.Globalization.CultureInfo.InvariantCulture);
                    tk[i].Slope.Y = Convert.ToDouble(lvAcceptedTracks.Items[i].SubItems[4].Text, System.Globalization.CultureInfo.InvariantCulture);
                }
                ef.Tracks = tk;
                ef.ShowDialog();
            }
        }

        private void btnSetPathsAndCSBrick_Click(object sender, EventArgs e)
        {
            int spn = 0;
            string sp = "";
            System.Collections.ArrayList zones = new System.Collections.ArrayList();
            foreach (ListViewItem lvi in lvAcceptedTracks.Items)
                if (lvi.Tag is CSBrickCand)
                {
                    spn++;
                    sp += ";\r\n" + (lvi.Tag as CSBrickCand).SetPathText;
                    zones.Add((lvi.Tag as CSBrickCand).IdZone);
                }
            foreach (CSBrickCand sz in CSBrickNotFound)
                if (zones.Contains(sz.IdZone) == false)
                {
                    spn++;
                    sp += ";\r\n" + sz.SetPathText;
                    zones.Add(sz.IdZone);
                }
            sp = "SetPaths " + spn.ToString() + sp;
            if (MessageBox.Show("Send interrupt?\r\n" + sp, "Interrupt prepared", MessageBoxButtons.OKCancel, MessageBoxIcon.Question) == DialogResult.OK)
            {
                SySal.OperaDb.OperaDbConnection conn = null;
                try
                {
                    SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
                    conn = cred.Connect();
                    conn.Open();
                    string bms = new SySal.OperaDb.OperaDbCommand("SELECT ADDRESS FROM TB_MACHINES WHERE ISBATCHSERVER > 0 AND ID_SITE = (SELECT TO_NUMBER(VALUE) FROM OPERA.LZ_SITEVARS WHERE NAME = 'ID_SITE')", conn).ExecuteScalar().ToString();
                    SySal.DAQSystem.BatchManager bm = (SySal.DAQSystem.BatchManager)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.BatchManager), "tcp://" + bms + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer).ToString() + "/BatchManager.rem");
                    bm.Interrupt(ProcOpId, cred.OPERAUserName, cred.OPERAPassword, sp);
                    MessageBox.Show("OK", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return;
                }
                catch (Exception x)
                {
                    MessageBox.Show("Error sending interrupt:\r\n" + x.ToString(), "Interrupt sending error", MessageBoxButtons.OK, MessageBoxIcon.Error);   
                }
                finally
                {
                    if (conn != null) conn.Close();
                }
            }
            if (MessageBox.Show("Save interrupt to file?", "Interrupt prepared", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
            {
                SaveFileDialog sdlg = new SaveFileDialog();
                sdlg.Title = "Select file for interrupt text";
                sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
                if (sdlg.ShowDialog() == DialogResult.OK)
                    try
                    {
                        System.IO.File.WriteAllText(sdlg.FileName, sp);
                        MessageBox.Show("OK", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    catch (Exception x)
                    {
                        MessageBox.Show("Error:\r\n" + x.ToString(), "File not written", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
            }
        }
    }
}