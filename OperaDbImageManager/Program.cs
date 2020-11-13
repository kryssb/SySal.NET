using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.OperaDbImageManager
{
    /// <summary>
    /// Allows uploading images to the DB and downloading images from the DB.    
    /// </summary>
    /// <remarks>
    /// <para>Images are stored in the <c>TB_EMULSION_IMAGES</c> table. This is linked to <c>TB_ZONES</c>, which in turn is linked to <c>TB_PROC_OPERATIONS</c>.</para>
    /// <para><b>Downloading images</b></para>
    /// <para>    
    /// In order to download images you must specify at least the brick and plate, or also the zone (optionally). 
    /// Then, all images linked to each zone in the specified set will be downloaded.
    /// The syntax is the following:
    /// <c>OperaDbImageManager /download &lt;path&gt; &lt;brick&gt; &lt;plate&gt; [zone]</c>
    /// as in the examples:
    /// <example><c>OperaDbImageManager /download c:\\temp\\ 1008622 54</c></example>
    /// <example><c>OperaDbImageManager /download c:\\temp\\ 1008622 54 1000010008888947</c></example>
    /// </para>
    /// <para><b>Uploading images</b></para>
    /// <para>Images can enter the DB only if information required about the brick, plate, machine, user, and settings used is filled.
    /// OperaDbImageManager performs all relevant tasks of creating process operation and zone; however, the information supplied must be 
    /// meaningful (i.e. the user, machine and configuration used must exist). If you don't have a configuration, please create one,
    /// filling the <c>SETTINGS</c> field with the relevant data taking parameters. While data are being uploaded, the operation number
    /// and zones are shown, so that they can be used later to publish the data set.
    /// The syntax to upload a set of files is the following:
    /// <c>OperaDbImageManager /upload &lt;brick&gt; &lt;plate&gt; &lt;series&gt; &lt;machine&gt; &lt;configuration&gt; &lt;minx&gt; &lt;maxx&gt; &lt;miny&gt; &lt;maxy&gt; &lt;txx&gt; &lt;txy&gt; &lt;tyx&gt; &lt;tyy&gt; &lt;tdx&gt; &lt;tdy&gt; &lt;calibrationop&gt; &lt;file1&gt; &lt;file2&gt; ...</c>
    /// </para>
    /// <para>It is possible to enter as many files as needed in a zone. Different zones must be entered by different executions of the program. 
    /// User information is taken from the user DB access credentials, which can be managed by <c>OperaDbGuiLogin</c>.
    /// The machine can be identified through its name or ID; the configuration can be identified through its name (use "" if the description contains blanks)
    /// or ID; if the calibration operation is set to 0 or to a negative number, the <c>CALIBRATION_OPERATION</c> is set to NULL.
    /// The following example shows how to enter 3 emulsion images.
    /// <example><c>OperaDbImageManager.exe /upload 1008622 54 3 mic3_sa 1000010008878963 10000 10300 20000 20300 1.01 0.001 -0.00099 1.001 15 -35 0 a.bmp b.bmp c.bmp</c></example>
    /// </para>
    /// </remarks>
    public class Exe
    {
        const string downloadstr = "/download";
        const string uploadstr = "/upload";

        static void Main(string[] args)
        {            
            try
            {
                SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
                SySal.OperaDb.OperaDbConnection Conn = cred.Connect();
                Conn.Open();
                if (String.Compare(args[0], downloadstr, true) == 0)
                {
                    int zones = 0;
                    int files = 0;
                    string basepath = args[1];
                    if (basepath.EndsWith("\\") == false && basepath.EndsWith("/") == false) basepath += "/";
                    long brick = Convert.ToInt64(args[2]);
                    int plate = Convert.ToInt32(args[3]);
                    long zone = 0;
                    long bytes = 0;
                    if (args.Length == 5) zone = Convert.ToInt64(args[4]);
                    System.Data.DataSet ds = new System.Data.DataSet();
                    string zonestr = "select id_eventbrick, id, series, minx, maxx, miny, maxy, txx, txy, tyx, tyy, tdx, tdy from tb_zones where id_eventbrick = " + brick + " and id_plate = " + plate;
                    if (zone > 0) zonestr += " and id = " + zone;
                    zonestr += " and exists (select * from tb_emulsion_images where tb_emulsion_images.id_eventbrick = tb_zones.id_eventbrick and tb_emulsion_images.id_zone = tb_zones.id)";
                    new SySal.OperaDb.OperaDbDataAdapter(zonestr, Conn).Fill(ds);
                    Console.WriteLine("Found " + ds.Tables[0].Rows.Count + " zone(s)");
                    Console.WriteLine("ID_EVENTBRICK\tID_ZONE\tSERIES\tMINX\tMAXX\tMINY\tMAXY\tTXX\tTXY\tTYX\tTYY\tTDX\tTDY\tFILE");
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    {
                        zones++;
                        SySal.OperaDb.OperaDbDataReader rdr = new SySal.OperaDb.OperaDbCommand("select id_zone, name, image_sequence_type, image_sequence from tb_emulsion_images where id_eventbrick = " + brick + " and id_zone = " + dr[1].ToString(), Conn).ExecuteReader();
                        while (rdr.Read())
                        {
                            Console.Write(brick + "\t" + dr[1].ToString() + "\t" + dr[2].ToString() + "\t" + dr[3].ToString() + "\t" + dr[4].ToString() + "\t" + dr[5].ToString() + "\t" + dr[6].ToString() + "\t" + dr[7].ToString() + "\t" + dr[8].ToString() + "\t" + dr[9].ToString() + "\t" + dr[10].ToString() + "\t" + dr[11].ToString() + "\t" + dr[12].ToString() + "\t");
                            files++;
                            byte[] b = (byte [])rdr.GetValue(3);
                            bytes += b.LongLength;
                            string name = dr[2].ToString() + "_" + rdr.GetValue(0).ToString() + "_" + rdr.GetValue(1).ToString() + "." + rdr.GetValue(2).ToString();
                            int lastslash = Math.Max(name.LastIndexOf('\\'), name.LastIndexOf('/'));
                            if (lastslash >= 0) name = name.Substring(lastslash + 1);
                            System.IO.File.WriteAllBytes(basepath + name, b);
                            Console.WriteLine(name);
                        }
                    }
                    Console.WriteLine("Downloaded " + zones + " zone(s), " + files + " file(s), " + bytes.ToString("N0", System.Globalization.CultureInfo.InvariantCulture) + " byte(s)");
                }
                else if (String.Compare(args[0], uploadstr, true) == 0)
                {
                    const int firstfilearg = 17;
                    long brick = Convert.ToInt64(args[1]);
                    int plate = Convert.ToInt32(args[2]);
                    long series = Convert.ToInt64(args[3]);
                    Console.WriteLine("Uploading data as " + cred.OPERAUserName);
                    long uid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM VW_USERS WHERE UPPER(USERNAME) = '" + cred.OPERAUserName.ToUpper() + "'", Conn).ExecuteScalar());
                    Console.WriteLine("User id = " + uid);
                    long machineid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_MACHINES WHERE UPPER(NAME) = '" + args[4].ToUpper() + "' OR TO_CHAR(ID) = '" + args[4] + "'", Conn).ExecuteScalar());
                    Console.WriteLine("Machine id = " + machineid);
                    long configid = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_PROGRAMSETTINGS WHERE UPPER(DESCRIPTION) = '" + args[5].ToUpper() + "' OR TO_CHAR(ID) = '" + args[5] + "'", Conn).ExecuteScalar());
                    Console.WriteLine("Configuration id = " + configid);                    
                    SySal.BasicTypes.Rectangle rect = new SySal.BasicTypes.Rectangle();
                    rect.MinX = Convert.ToDouble(args[6], System.Globalization.CultureInfo.InvariantCulture);
                    rect.MaxX = Convert.ToDouble(args[7], System.Globalization.CultureInfo.InvariantCulture);
                    rect.MinY = Convert.ToDouble(args[8], System.Globalization.CultureInfo.InvariantCulture);
                    rect.MaxY = Convert.ToDouble(args[9], System.Globalization.CultureInfo.InvariantCulture);
                    double txx = Convert.ToDouble(args[10], System.Globalization.CultureInfo.InvariantCulture);
                    double txy = Convert.ToDouble(args[11], System.Globalization.CultureInfo.InvariantCulture);
                    double tyx = Convert.ToDouble(args[12], System.Globalization.CultureInfo.InvariantCulture);
                    double tyy = Convert.ToDouble(args[13], System.Globalization.CultureInfo.InvariantCulture);
                    double tdx = Convert.ToDouble(args[14], System.Globalization.CultureInfo.InvariantCulture);
                    double tdy = Convert.ToDouble(args[15], System.Globalization.CultureInfo.InvariantCulture);
                    SySal.OperaDb.Schema.DB = Conn; 
                    SySal.OperaDb.OperaDbTransaction trans = Conn.BeginTransaction();
                    long procopid = 0;
                    object po = new object();
                    long calibop = Convert.ToInt64(args[16]);
                    SySal.OperaDb.Schema.PC_ADD_PROC_OPERATION_PLATE.Call(machineid, configid, uid, brick, plate, null, ((calibop < 0) ? (object)null : (object)calibop), System.DateTime.Now, null, ref po);
                    procopid = SySal.OperaDb.Convert.ToInt64(po);
                    Console.WriteLine("Process Operation: " + procopid);
                    System.DateTime starttime = System.IO.File.GetCreationTime(args[firstfilearg]);
                    System.DateTime endtime = starttime;
                    int i;
                    for (i = (firstfilearg + 1); i < args.Length; i++)
                    {
                        System.DateTime ftime = System.IO.File.GetLastAccessTime(args[i]);
                        if (ftime < starttime) starttime = ftime;
                        else if (ftime > endtime) endtime = ftime;
                    }
                    long zoneid = SySal.OperaDb.Schema.TB_ZONES.Insert(brick, plate, procopid, 0, rect.MinX, rect.MaxX, rect.MinY, rect.MaxY, "Images", starttime, endtime, series, txx, txy, tyx, tyy, tdx, tdy);
                    Console.WriteLine("Zone Id: " + zoneid);
                    SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_EMULSION_IMAGES (ID_EVENTBRICK, ID_ZONE, NAME, IMAGE_SEQUENCE_TYPE, IMAGE_SEQUENCE) VALUES (:idb, :idz, :nm, :ist, :iss)", Conn, trans);
                    cmd.Parameters.Add("idb", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = brick;
                    cmd.Parameters.Add("idz", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input).Value = zoneid;
                    cmd.Parameters.Add("nm", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
                    cmd.Parameters.Add("ist", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
                    cmd.Parameters.Add("iss", SySal.OperaDb.OperaDbType.BLOB, System.Data.ParameterDirection.Input);
                    for (i = firstfilearg; i < args.Length; i++)
                    {
                        string path = args[i];
                        if (path.Length > 255) path = path.Substring(path.Length - 255);
                        string type = path.Substring(path.LastIndexOf('.') + 1).ToUpper();
                        cmd.Parameters[2].Value = path.Substring(0, path.LastIndexOf('.'));
                        cmd.Parameters[3].Value = type;
                        cmd.Parameters[4].Value = System.IO.File.ReadAllBytes(args[i]);
                        cmd.ExecuteNonQuery();
                        Console.WriteLine("Written " + args[i]);
                    }
                    trans.Commit();
                }
                else throw new Exception();
            }
            catch (Exception x)
            {
                if (x.Message.Length > 0) Console.WriteLine(x.ToString());
                Console.WriteLine("usage: OperaDbImageManager /upload <brick> <plate> <series> <machine> <configuration> <minx> <maxx> <miny> <maxy> <txx> <txy> <tyx> <tyy> <tdx> <tdy> <calibrationop> <file1> <file2> ...");
                Console.WriteLine("usage: OperaDbImageManager /download <path> <brick> <plate> [zone]");
            }
        }
    }
}
