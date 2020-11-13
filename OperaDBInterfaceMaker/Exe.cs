using System;

namespace SySal.Executables.OperaDBInterfaceMaker
{
	/// <summary>
	/// OperaDBInterfaceMaker.
	/// </summary>
	/// <remarks>
	/// <para>This is an utility executable. It is intended to run very seldom, normally for code development purposes. It is automatically run on build success, to generate updated DB <see cref="SySal.OperaDb.Schema">schema</see> classes.</para>
	/// <para>OperaDBInterfaceMaker reads metadata views (the <c>ALL_...</c> views in the SYS account) to browse the OPERA schema and generate interface classes for tables and procedures.</para>
	/// <para>The output is normally sent to the output stream (the console), but it is a common practice to redirect it to a file in C#, which can be then compiled.</para>
	/// <para>The executable receives no parameters. The DB is browsed by using the current default credential record.</para>
	/// </remarks>
	public class Exe
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
			conn.Open();
			Console.WriteLine("namespace SySal.OperaDb");
			Console.WriteLine("{");
			Console.WriteLine("\t/// <summary>");
			Console.WriteLine("\t/// Static class that allows access to schema components of the OPERA DB.");
			Console.WriteLine("\t/// Its DB field must be set before the class can be used.");
			Console.WriteLine("\t/// </summary>");
			Console.WriteLine("\tpublic class Schema");
			Console.WriteLine("\t{");
			Console.WriteLine("\t\tprivate Schema() {}");
			Console.WriteLine("\t\t/// <summary>");
			Console.WriteLine("\t\t/// Retrieval order for query results.");
			Console.WriteLine("\t\t/// </summary>");
			Console.WriteLine("\t\tpublic enum OrderBy { None, Ascending, Descending }");
			string ondbsetprepared = "";
			System.Data.DataSet ds_tb = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT TABLE_NAME FROM ALL_TABLES WHERE OWNER = 'OPERA' AND (TABLE_NAME LIKE 'TB_%' OR TABLE_NAME LIKE 'LZ_%') ORDER BY TABLE_NAME ASC", conn, null).Fill(ds_tb);
			foreach (System.Data.DataRow dr_tb in ds_tb.Tables[0].Rows)
			{				
				ondbsetprepared += "\t\t\t\tif (" + dr_tb[0].ToString() + ".Prepared) { " + dr_tb[0].ToString() + ".cmd.Connection = null; " + dr_tb[0].ToString() + ".Prepared = false; }\n";
				string AutoIdColumn = null;
				try
				{
					AutoIdColumn = new SySal.OperaDb.OperaDbCommand("SELECT COLUMN_NAME FROM ALL_TRIGGER_COLS WHERE TABLE_NAME = '" + dr_tb[0].ToString() + "' AND TABLE_OWNER = 'OPERA' AND COLUMN_USAGE = 'NEW OUT'", conn, null).ExecuteScalar().ToString();
					if (AutoIdColumn == "") AutoIdColumn = null;
				}
				catch (Exception)
				{
					AutoIdColumn = null;
				}
				Console.WriteLine("\t\t/// <summary>");
				Console.WriteLine("\t\t/// Accesses the " + dr_tb[0].ToString() + " table in the DB.");
				if (AutoIdColumn == null)
					Console.WriteLine("\t\t/// For data insertion, the Insert and Flush static methods are used. This class inserts data in batches.");
				else
					Console.WriteLine("\t\t/// For data insertion, the Insert method is used. Rows are inserted one by one.");
				Console.WriteLine("\t\t/// An instance of the class is produced for data retrieval.");
				Console.WriteLine("\t\t/// </summary>");
				Console.WriteLine("\t\tpublic class " + dr_tb[0].ToString());
				Console.WriteLine("\t\t{");
				Console.WriteLine("\t\t\tinternal " + dr_tb[0].ToString() + "() {}");
				Console.WriteLine("\t\t\tSystem.Data.DataRowCollection m_DRC;");
				if (AutoIdColumn == null)
				{
					Console.WriteLine("\t\t\tconst int ArraySize = 100;");
					Console.WriteLine("\t\t\tstatic int index = 0;");
					Console.WriteLine("\t\t\t/// <summary>");
					Console.WriteLine("\t\t\t/// Since this class uses batch insertion, this method must be called to ensure that all rows are actually inserted into " + dr_tb[0].ToString() + ". Failure to do so will result in incomplete writes.");
					Console.WriteLine("\t\t\t/// </summary>");					
					Console.WriteLine("\t\t\tstatic public void Flush() { if (cmd == null || index == 0) return; cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }");
				};
				System.Data.DataSet ds_col = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT COLUMN_NAME, DATA_TYPE, DATA_PRECISION, DATA_SCALE, NULLABLE, DATA_LENGTH FROM ALL_TAB_COLUMNS WHERE OWNER = 'OPERA' AND TABLE_NAME = '" + dr_tb[0].ToString() + "' ORDER BY COLUMN_ID", conn, null).Fill(ds_col);
				object pk_obj = new SySal.OperaDb.OperaDbCommand("SELECT CONSTRAINT_NAME FROM ALL_CONSTRAINTS WHERE OWNER = 'OPERA' AND CONSTRAINT_TYPE = 'P' AND TABLE_NAME = '" + dr_tb[0].ToString() + "'", conn, null).ExecuteScalar();				
				string colstring = "";
				string parstring = "";
				string parslotstring = "";
				int parcount = 0;
				string insertparamstring = "";
				string insertparslotstring = "";
				string inserthelpparamstring = "";				
				foreach (System.Data.DataRow dr_col in ds_col.Tables[0].Rows)
				{
					if (parcount > 0)
					{
						colstring += ",";
						parstring += ",";
						insertparamstring += ",";
					}
					parcount++;
					colstring += dr_col[0].ToString();
					parstring += ":p_" + parcount;
					
					string array_type = "object";
					string data_type = "UNKNOWN-" + dr_col[1].ToString();
                    string oratype_string = "UNKNOWN-" + dr_col[1].ToString();
					string convertstr = "UNKNOWN";
					bool isnullable = true;
					if (dr_col[1].ToString() == "NUMBER" && dr_col[3].ToString() == "0" )
					{						
						if (dr_col[2] == System.DBNull.Value || Convert.ToInt32(dr_col[2]) > 8)
						{
							data_type = "long";
							oratype_string = "SySal.OperaDb.OperaDbType.Long";
							convertstr = "Int64";
						}
						else
						{
							data_type = "int";
							oratype_string = "SySal.OperaDb.OperaDbType.Int";
							convertstr = "Int32";
						}
					}
					else if (dr_col[1].ToString() == "FLOAT" || dr_col[1].ToString() == "NUMBER")
					{
						data_type = "double";
						oratype_string = "SySal.OperaDb.OperaDbType.Double";
						convertstr = "Double";
					}
					else if (dr_col[1].ToString() == "CHAR")
					{
						if (SySal.OperaDb.Convert.ToInt32(dr_col[5]) == 1)
						{
							data_type = "char";		
							oratype_string = "SySal.OperaDb.OperaDbType.String";
							convertstr = "Char";
						}
						else
						{
							data_type = "string";		
							oratype_string = "SySal.OperaDb.OperaDbType.String";
							convertstr = "String";
						}
					}
					else if (dr_col[1].ToString() == "VARCHAR2")
					{
						data_type = "string";
						oratype_string = "SySal.OperaDb.OperaDbType.String";
						convertstr = "String";
					}
					else if (dr_col[1].ToString() == "DATE" || dr_col[1].ToString().StartsWith("TIMESTAMP"))
					{
						data_type = "System.DateTime";
						oratype_string = "SySal.OperaDb.OperaDbType.DateTime";
						convertstr = "DateTime";
					}
					else if (dr_col[1].ToString() == "CLOB")
					{
						data_type = "string";
						oratype_string = "SySal.OperaDb.OperaDbType.CLOB";
						convertstr = "String";
					}
                    else if (dr_col[1].ToString() == "BLOB")
                    {
                        data_type = "byte []";
                        oratype_string = "SySal.OperaDb.OperaDbType.BLOB";
                        convertstr = "Bytes";
                    }
                    if (dr_col[4].ToString() == "N") { array_type = data_type; isnullable = false; }
					parslotstring += "\t\t\t\tnewcmd.Parameters.Add(\"p_" + parcount + "\", " + oratype_string + ", System.Data.ParameterDirection.Input)" + ((AutoIdColumn != null) ? "" : (".Value = a_" + dr_col[0].ToString().Replace("#", "_SHARP_") )) + ";\n";
                    insertparamstring += array_type + " i_" + dr_col[0].ToString().Replace("#", "_SHARP_");
					if (AutoIdColumn == null || AutoIdColumn != dr_col[0].ToString())
                        inserthelpparamstring += "\t\t\t/// <param name=\"i_" + dr_col[0].ToString().Replace("#", "_SHARP_") + "\">the value to be inserted for " + dr_col[0].ToString() + (isnullable ? (". The value for this parameter can be " + data_type + " or System.DBNull.Value") : "") + ".</param>\n";
					else
                        inserthelpparamstring += "\t\t\t/// <param name=\"i_" + dr_col[0].ToString().Replace("#", "_SHARP_") + "\">the value to be inserted for " + dr_col[0].ToString() + (isnullable ? (". The value for this parameter can be " + data_type + " or System.DBNull.Value") : "") + ". This value is actually used only if this method call is involved in data publication/replication, otherwise the actual value is generated by the OPERA DB and the supplied value is ignored.</param>\n";
					if (AutoIdColumn == null)
					{
                        insertparslotstring += "\t\t\t\ta_" + dr_col[0].ToString().Replace("#", "_SHARP_") + "[index] = i_" + dr_col[0].ToString().Replace("#", "_SHARP_") + ";\n";
                        Console.WriteLine("\t\t\tprivate static " + array_type + " [] a_" + dr_col[0].ToString().Replace("#", "_SHARP_") + " = new " + array_type + "[ArraySize];");
					}
					else
					{
                        insertparslotstring += "\t\t\t\tcmd.Parameters[" + (parcount - 1) + "].Value = i_" + dr_col[0].ToString().Replace("#", "_SHARP_") + ";\n";
					}

					Console.WriteLine("\t\t\t/// <summary>");
					Console.WriteLine("\t\t\t/// Retrieves " + dr_col[0].ToString() + " for the current row." + (isnullable ? (" The return value can be System.DBNull.Value or a value that can be cast to " + data_type + ".") : ""));
					Console.WriteLine("\t\t\t/// </summary>");
                    Console.WriteLine("\t\t\tpublic " + array_type + " _" + dr_col[0].ToString().Replace("#", "_SHARP_"));
					Console.WriteLine("\t\t\t{");
					Console.WriteLine("\t\t\t\tget");
					Console.WriteLine("\t\t\t\t{");
					if (isnullable) Console.WriteLine("\t\t\t\t\tif (m_DR[" + (parcount - 1) + "] == System.DBNull.Value) return System.DBNull.Value;");
                    if (convertstr == "Bytes") Console.WriteLine("\t\t\t\t\treturn SySal.OperaDb.Convert.To" + convertstr + "(m_DR[" + (parcount - 1) + "]);");
                    else Console.WriteLine("\t\t\t\t\treturn System.Convert.To" + convertstr + "(m_DR[" + (parcount - 1) + "]);");
					Console.WriteLine("\t\t\t\t}");
					Console.WriteLine("\t\t\t}");
				}
				if (AutoIdColumn != null)
				{
					parslotstring += "\t\t\t\tnewcmd.Parameters.Add(\"o_" + parcount + "\", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);\n";

					Console.WriteLine("\t\t\t/// <summary>");
					Console.WriteLine("\t\t\t/// Inserts one row into the DB. The row is inserted immediately.");
					Console.WriteLine("\t\t\t/// </summary>");			
					Console.Write(inserthelpparamstring);
					Console.WriteLine("\t\t\t/// <returns>the value of " + AutoIdColumn + " for the new row.</returns>");
					Console.WriteLine("\t\t\tstatic public long Insert(" + insertparamstring + ")");
					Console.WriteLine("\t\t\t{");
					Console.WriteLine("\t\t\t\tif (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };");
					Console.Write(insertparslotstring);
					Console.WriteLine("\t\t\t\tcmd.ExecuteNonQuery();");
					Console.WriteLine("\t\t\t\treturn SySal.OperaDb.Convert.ToInt64(cmd.Parameters[" + parcount +"].Value);");
					Console.WriteLine("\t\t\t}");
				}
				else
				{
					Console.WriteLine("\t\t\t/// <summary>");
					Console.WriteLine("\t\t\t/// Inserts one row into the DB. The row is actually inserted later, in a batch insertion command. The Flush method must be called to ensure all rows are actually written.");
					Console.WriteLine("\t\t\t/// </summary>");					
					Console.Write(inserthelpparamstring);
					Console.WriteLine("\t\t\tstatic public void Insert(" + insertparamstring + ")");
					Console.WriteLine("\t\t\t{");
					Console.WriteLine("\t\t\t\tif (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };");
					Console.Write(insertparslotstring);
					Console.WriteLine("\t\t\t\tif (++index >= ArraySize) { cmd.ArrayBindCount = index; cmd.ExecuteNonQuery(); index = 0; }");
					Console.WriteLine("\t\t\t}");
				}

				if (pk_obj != null && pk_obj != System.DBNull.Value)
				{
					System.Data.DataSet dpks = new System.Data.DataSet();
					new SySal.OperaDb.OperaDbDataAdapter("select column_name from all_cons_columns where all_cons_columns.owner = 'OPERA' and constraint_name = '" + pk_obj.ToString() + "'", conn, null).Fill(dpks);
					string selpkparams = "";
					string orderbyascstring = "";
					string orderbydescstring = "";
					string selecthelpparamstring = "";
					foreach (System.Data.DataRow dpkr in dpks.Tables[0].Rows)
					{
						if (selpkparams.Length == 0)
						{
							selpkparams = "object i_" + dpkr[0].ToString().Replace("#", "_SHARP_");
							orderbyascstring = dpkr[0].ToString() + " ASC";
							orderbydescstring = dpkr[0].ToString() + " DESC";							
						}
						else
						{
                            selpkparams += ",object i_" + dpkr[0].ToString().Replace("#", "_SHARP_");
							orderbyascstring += "," + dpkr[0].ToString() + " ASC";
							orderbydescstring += "," + dpkr[0].ToString() + " DESC";
						}
                        selecthelpparamstring += "\t\t\t/// <param name=\"i_" + dpkr[0].ToString().Replace("#", "_SHARP_") + "\">if non-null, only rows that have this field equal to the specified value are returned.</param>\n";
					}
					Console.WriteLine("\t\t\t/// <summary>");
					Console.WriteLine("\t\t\t/// Reads a set of rows from " + dr_tb[0].ToString() + " and retrieves them into a new " + dr_tb[0].ToString() + " object.");
					Console.WriteLine("\t\t\t/// </summary>");
					Console.Write(selecthelpparamstring);
					Console.WriteLine("\t\t\t/// <param name=\"order\">the ordering scheme to be applied to returned rows." + ((dpks.Tables[0].Rows.Count > 1) ? " This applies to all columns in the primary key." : "")+ "</param>");
					Console.WriteLine("\t\t\t/// <returns>a new instance of the " + dr_tb[0].ToString() + " class that can be used to read the retrieved data.</returns>");
					Console.WriteLine("\t\t\tstatic public " + dr_tb[0].ToString() + " SelectPrimaryKey(" + selpkparams + ", OrderBy order)");
					Console.WriteLine("\t\t\t{");
					Console.WriteLine("\t\t\t\tstring wherestr = \"\";");
					Console.WriteLine("\t\t\t\tstring wtempstr = \"\";");
					foreach (System.Data.DataRow dpkr in dpks.Tables[0].Rows)
					{
                        Console.WriteLine("\t\t\t\tif (i_" + dpkr[0].ToString().Replace("#", "_SHARP_") + " != null)");
						Console.WriteLine("\t\t\t\t{");
                        Console.WriteLine("\t\t\t\t\tif (i_" + dpkr[0].ToString().Replace("#", "_SHARP_") + " == System.DBNull.Value) wtempstr = \"" + dpkr[0] + " IS NULL\";");
                        Console.WriteLine("\t\t\t\t\telse wtempstr = \"" + dpkr[0] + " = \" + i_" + dpkr[0].ToString().Replace("#", "_SHARP_") + ".ToString();");
						Console.WriteLine("\t\t\t\t\tif (wherestr.Length == 0) wherestr = wtempstr; else wherestr += \" AND \" + wtempstr;");
						Console.WriteLine("\t\t\t\t}");
					}
					Console.WriteLine("\t\t\t\tif (order == OrderBy.Ascending) return SelectWhere(wherestr, \"" + orderbyascstring + "\");");
					Console.WriteLine("\t\t\t\telse if (order == OrderBy.Descending) return SelectWhere(wherestr, \"" + orderbydescstring + "\");");
					Console.WriteLine("\t\t\t\treturn SelectWhere(wherestr, null);");
					Console.WriteLine("\t\t\t}");
				}

				Console.WriteLine("\t\t\t/// <summary>");
				Console.WriteLine("\t\t\t/// Reads a set of rows from " + dr_tb[0].ToString() + " and retrieves them into a new " + dr_tb[0].ToString() + " object.");
				Console.WriteLine("\t\t\t/// </summary>");
				Console.WriteLine("\t\t\t/// <param name=\"wherestr\">the string to be used in the WHERE clause. If null or empty, no WHERE clause is generated and all rows are returned.</param>");
				Console.WriteLine("\t\t\t/// <param name=\"orderstr\">the string to be used in the ORDER BY clause. If null or empty, no ORDER BY clause is generated and rows are returned in an unspecified, non-deterministic order.</param>");
				Console.WriteLine("\t\t\t/// <returns>a new instance of the " + dr_tb[0].ToString() + " class that can be used to read the retrieved data.</returns>");
				Console.WriteLine("\t\t\tstatic public " + dr_tb[0].ToString() + " SelectWhere(string wherestr, string orderstr)");
				Console.WriteLine("\t\t\t{");
				Console.WriteLine("\t\t\t\t" + dr_tb[0].ToString() + " newobj = new " + dr_tb[0].ToString() + "();");
				Console.WriteLine("\t\t\t\tSystem.Data.DataSet ds = new System.Data.DataSet();");
				Console.WriteLine("\t\t\t\tnew SySal.OperaDb.OperaDbDataAdapter(\"SELECT " + colstring + " FROM " + dr_tb[0] + "\" + ((wherestr == null || wherestr.Trim().Length == 0) ? \"\" : (\" WHERE(\" + wherestr + \")\" + ((orderstr != null && orderstr.Trim() != \"\") ? (\" ORDER BY \" + orderstr): \"\") )), Schema.m_DB).Fill(ds);");
				Console.WriteLine("\t\t\t\tnewobj.m_DRC = ds.Tables[0].Rows;");
				Console.WriteLine("\t\t\t\tnewobj.m_Row = -1;");
				Console.WriteLine("\t\t\t\treturn newobj;");
				Console.WriteLine("\t\t\t}");
				Console.WriteLine("\t\t\t/// <summary>");
				Console.WriteLine("\t\t\t/// the number of rows retrieved.");
				Console.WriteLine("\t\t\t/// </summary>");
				Console.WriteLine("\t\t\tpublic int Count { get { return m_DRC.Count; } }");
				Console.WriteLine("\t\t\tinternal int m_Row;");
				Console.WriteLine("\t\t\tinternal System.Data.DataRow m_DR;");
				Console.WriteLine("\t\t\t/// <summary>");
				Console.WriteLine("\t\t\t/// the current row for which field values are exposed.");
				Console.WriteLine("\t\t\t/// </summary>");
				Console.WriteLine("\t\t\tpublic int Row { get { return m_Row; } set { m_Row = value; m_DR = m_DRC[m_Row]; } }");

				Console.WriteLine("\t\t\tinternal static bool Prepared = false;");
				Console.WriteLine("\t\t\tinternal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();");
				Console.WriteLine("\t\t\tprivate static SySal.OperaDb.OperaDbCommand InitCommand()");
				Console.WriteLine("\t\t\t{");
				Console.WriteLine("\t\t\t\tSySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand(\"INSERT INTO " + dr_tb[0].ToString() + " (" + colstring + ") VALUES (" + parstring + ")" + ((AutoIdColumn != null) ? (" RETURNING " + AutoIdColumn + " INTO :o_" + parcount) : "") + "\");");
				Console.Write(parslotstring);
				Console.WriteLine("\t\t\t\treturn newcmd;");
				Console.WriteLine("\t\t\t}");
				Console.WriteLine("\t\t}");
			}

			System.Data.DataSet ds_pc = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT OBJECT_NAME FROM ALL_PROCEDURES WHERE OWNER = 'OPERA' AND (OBJECT_NAME LIKE 'PC_%' OR OBJECT_NAME LIKE 'LP_%') ORDER BY OBJECT_NAME ASC", conn, null).Fill(ds_pc);
			foreach (System.Data.DataRow dr_pc in ds_pc.Tables[0].Rows)
			{
				ondbsetprepared += "\t\t\t\tif (" + dr_pc[0].ToString() + ".Prepared) { " + dr_pc[0].ToString() + ".cmd.Connection = null; " + dr_pc[0].ToString() + ".Prepared = false; }\n";
				Console.WriteLine("\t\t/// <summary>");
				Console.WriteLine("\t\t/// Accesses the " + dr_pc[0].ToString() + " procedure in the DB.");
				Console.WriteLine("\t\t/// </summary>");
				Console.WriteLine("\t\tpublic class " + dr_pc[0].ToString());
				Console.WriteLine("\t\t{");
				Console.WriteLine("\t\t\tprivate " + dr_pc[0].ToString() + "() {}");
				System.Data.DataSet ds_arg = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ARGUMENT_NAME, IN_OUT, DATA_TYPE FROM ALL_ARGUMENTS WHERE OWNER = 'OPERA' AND OBJECT_NAME = '" + dr_pc[0].ToString() + "' ORDER BY POSITION ASC", conn, null).Fill(ds_arg);
				string argstr = "";
				string oratype_string = "";
				string parslotstring = "";
				string callslotstring = "";
				string callargstring = "";
				string paramretstring = "";
				string paramhelpstring = "";
				int parcount = 0;
				foreach (System.Data.DataRow dra in ds_arg.Tables[0].Rows)
				{
					parcount++;
					bool isout = false;
					bool isinout = false;
					if (dra[1].ToString() == "OUT")
					{
						isout = true;
						paramretstring += "\t\t\t\tif (" + dra[0].ToString() + " != null) " + dra[0].ToString() + " = cmd.Parameters[" + (parcount - 1).ToString() + "].Value;\n";
						paramhelpstring += "\t\t\t/// <param name = \"" + dra[0].ToString() + "\">the value of " + dra[0].ToString() + " obtained from the procedure call as an output. If null, the output is ignored.</param>\n";
					}
					else if (dra[1].ToString() == "IN/OUT")
					{
						isinout = true;
						paramretstring += "\t\t\t\tif (" + dra[0].ToString() + " != null) " + dra[0].ToString() + " = cmd.Parameters[" + (parcount - 1).ToString() + "].Value;\n";
						paramhelpstring += "\t\t\t/// <param name = \"" + dra[0].ToString() + "\">the value of " + dra[0].ToString() + " to be used for the procedure call, which can be modified on procedure completion as an output. If null, the input is replaced with a <c>System.DBNull.Value</c> and the output is ignored.</param>\n";
					}
					else if (dra[1].ToString() == "IN")
					{
						paramhelpstring += "\t\t\t/// <param name = \"" + dra[0].ToString() + "\">the value of " + dra[0].ToString() + " to be used for the procedure call.</param>\n";
					}
					else throw new Exception("Unsupported parameter direction " + dra[1].ToString());
					if (dra[2].ToString() == "NUMBER")
					{						
						oratype_string = "SySal.OperaDb.OperaDbType.Long";
					}
					else if (dra[2].ToString() == "FLOAT")
					{
						oratype_string = "SySal.OperaDb.OperaDbType.Double";
					}
					else if (dra[2].ToString() == "CHAR")
					{
						oratype_string = "SySal.OperaDb.OperaDbType.String";
					}
					else if (dra[2].ToString() == "VARCHAR2")
					{
						oratype_string = "SySal.OperaDb.OperaDbType.String";
					}
					else if (dra[2].ToString() == "DATE" || dra[2].ToString().StartsWith("TIMESTAMP"))
					{
						oratype_string = "SySal.OperaDb.OperaDbType.DateTime";
					}
					else if (dra[2].ToString() == "CLOB")
					{
						oratype_string = "SySal.OperaDb.OperaDbType.CLOB";
					}
					else oratype_string = "UNKNOWN-" + dra[2].ToString();
					if (argstr == "")
					{
						argstr = (isout ? "ref " : "") + ("object " + dra[0].ToString());
						callargstring = ":p_" + parcount.ToString();
					}
					else
					{
						argstr += ", " + (isout ? "ref " : "") + ("object " + dra[0].ToString());
						callargstring += ",:p_" + parcount.ToString();
					}
					parslotstring += "\t\t\t\tnewcmd.Parameters.Add(\"p_" + parcount + "\", " + oratype_string + ", System.Data.ParameterDirection." + (isout ? "Output" : (isinout ? "InputOutput" : "Input")) + ");\n";
					if (!isout)	
					{
						if (isinout) callslotstring += "\t\t\t\tif (" + dra[0].ToString() + " == null) cmd.Parameters[" + (parcount - 1) + "].Value = System.DBNull.Value; else cmd.Parameters[" + (parcount - 1) + "].Value = " + dra[0].ToString() + ";\n";						
						else callslotstring += "\t\t\t\tcmd.Parameters[" + (parcount - 1) + "].Value = " + dra[0].ToString() + ";\n";
					}
				}
				Console.WriteLine("\t\t\tinternal static bool Prepared = false;");
				Console.WriteLine("\t\t\tinternal static SySal.OperaDb.OperaDbCommand cmd = InitCommand();");
				Console.WriteLine("\t\t\tprivate static SySal.OperaDb.OperaDbCommand InitCommand()");
				Console.WriteLine("\t\t\t{");
				Console.WriteLine("\t\t\t\tSySal.OperaDb.OperaDbCommand newcmd = new SySal.OperaDb.OperaDbCommand(\"CALL " + dr_pc[0].ToString() + "(" + callargstring + ")\");");
				Console.Write(parslotstring);
				Console.WriteLine("\t\t\t\treturn newcmd;");
				Console.WriteLine("\t\t\t}");
				Console.WriteLine("\t\t\t/// <summary>");
				Console.WriteLine("\t\t\t/// Calls the procedure.");
				Console.WriteLine("\t\t\t/// </summary>");
				Console.Write(paramhelpstring);
				Console.WriteLine("\t\t\tpublic static void Call(" + argstr + ")");
				Console.WriteLine("\t\t\t{");
				Console.WriteLine("\t\t\t\tif (!Prepared) { cmd.Connection = Schema.m_DB; Prepared = true; cmd.Prepare(); };");
				Console.Write(callslotstring);
				Console.WriteLine("\t\t\t\tcmd.ExecuteNonQuery();");
				Console.Write(paramretstring);
				Console.WriteLine("\t\t\t}");
				Console.WriteLine("\t\t}");
			}

			Console.WriteLine("\t\tstatic internal SySal.OperaDb.OperaDbConnection m_DB = null;");
			Console.WriteLine("\t\t/// <summary>");
			Console.WriteLine("\t\t/// The DB connection currently used by the Schema class. Must be set before using any child class.");
			Console.WriteLine("\t\t/// </summary>");
			Console.WriteLine("\t\tstatic public SySal.OperaDb.OperaDbConnection DB");
			Console.WriteLine("\t\t{");
			Console.WriteLine("\t\t\tget { return m_DB; }");
			Console.WriteLine("\t\t\tset");
			Console.WriteLine("\t\t\t{");
			Console.Write(ondbsetprepared);
			Console.WriteLine("\t\t\t\tm_DB = value;");
			Console.WriteLine("\t\t\t}");
			Console.WriteLine("\t\t}");
			Console.WriteLine("\t}");
			Console.WriteLine("}");
		}
	}
}
