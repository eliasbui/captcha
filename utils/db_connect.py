import os
import sqlite3
import datetime

def make_list_insert(list_first, list_second):
  list_texts = []
  for i in range(len(list_first)):
    list_texts.append(
      "(\"%s\",\"%s\",\"%s\")" % (list_first[i], list_second[i], str(datetime.datetime.now()))
    )
  return list_texts

class LogDataBase:
  def __init__(self) -> None:
    if not os.path.exists("./captcha_images.db"):
      self.con = sqlite3.connect("captcha_images.db",
              detect_types=sqlite3.PARSE_DECLTYPES |
                          sqlite3.PARSE_COLNAMES)
      self.create_table()
      print("Create new database")
    else:
      self.con = sqlite3.connect("captcha_images.db",
              detect_types=sqlite3.PARSE_DECLTYPES |
                          sqlite3.PARSE_COLNAMES)
    
  def create_table(self):
    cur = self.con.cursor()
    cur.execute("CREATE TABLE image_source \
      (image_id TEXT PRIMARY KEY, \
        image_path TEXT NOT NULL, \
          time_create timestamp); \
    ")
    
    cur.execute("CREATE TABLE image_pred \
      (image_id TEXT PRIMARY KEY,\
        pred_text TEXT NOT NULL, \
          time_create timestamp) \
    ")
    
    cur.execute("CREATE TABLE image_label\
      (image_id TEXT PRIMARY KEY, \
        label_text TEXT NOT NULL, \
          time_create timestamp)\
    ")
    self.con.commit()

  def add_images(self, list_id_images, list_image_path):
    cur = self.con.cursor()
    form_add_image = "INSERT INTO image_source VALUES "
    list_texts = make_list_insert(list_id_images, list_image_path)
    form_add_image += ", ".join(list_texts)
    cur.execute(form_add_image)
    cur.close()
    self.con.commit()

  def add_preds(self, list_id_images, list_image_preds):
    cur = self.con.cursor()
    form_add_image = "INSERT INTO image_pred VALUES "
    list_texts = make_list_insert(list_id_images, list_image_preds)
    form_add_image += ", ".join(list_texts)
    cur.execute(form_add_image)
    cur.close()
    self.con.commit()

  def add_labels(self, list_id_images, list_image_labels):
    cur = self.con.cursor()
    form_add_image = "INSERT INTO image_pred VALUES "
    list_texts = make_list_insert(list_id_images, list_image_labels)
    form_add_image += ", ".join(list_texts)
    cur.execute(form_add_image)
    cur.close()
    self.con.commit()

  def check_idx(self, list_image_id):
    cur = self.con.cursor()
    form_check = "SELECT image_id FROM\
      image_source WHERE image_id IN \
        (%s)" % (", ".join(
          ["\"" + elm + "\"" for elm in list_image_id]
        ))
    cur.execute(form_check)
    list_image_id = cur.fetchall()
    cur.close()
    return list_image_id

  def training_take_ids(self):
    cur = self.con.cursor()
    # take index from image_label
    form_check = "SELECT image_id FROM image_label \
      where image_id in \
        (SELECT image_id FROM image_source)"
    cur.execute(form_check)
    list_image_id = cur.fetchall()
    cur.close()
    return list_image_id
  
  def training_take_labels(self, list_image_id):
    cur = self.con.cursor()
    form_check = "SELECT label_text FROM image_label \
      where image_id in (%s)" % (", ".join(
        ["\"" + elm + "\"" for elm in list_image_id]
      ))
    cur.execute(form_check)
    list_image_label = cur.fetchall()
    cur.close()
    return list_image_label
  
  def training_take_paths(self, list_image_id):
    cur = self.con.cursor()
    form_check = "SELECT image_path FROM image_source \
      where image_id in (%s)" % (", ".join(
        ["\"" + elm + "\"" for elm in list_image_id]
      ))
    cur.execute(form_check)
    list_image_path = cur.fetchall()
    cur.close()
    return list_image_path
  